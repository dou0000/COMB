from __future__ import annotations
import math
import numbers
import typing as t
from typing import Type, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReflectionPad2d, ReplicationPad2d, ConstantPad2d

from models.networks.base_network import BaseNetwork

Tensor = torch.Tensor


# ============================================================
# AnchorBankChannelAttention (unchanged full lattice, fast)
# ============================================================

class AnchorBankChannelAttention(nn.Module):
    """
    Channel Attention (CBAM-style) with a per-(y,x) anchor bank.

    - gain_bank: [Y, X, C] on store_device
    - init_mask: [Y, X] bool
    - neigh_cache[y][x]: neighbor coordinates & distances as Tensor[K,3]

    This is already vectorized and fast; we don't memory-bound CBAM,
    because its footprint is tiny compared to halo banks.
    """

    def __init__(self,
                 in_ch: int,
                 r: int = 16,
                 *,
                 alpha: float = 0.10,
                 prefill_m: float = 1.0,
                 include_center: bool = True,
                 neighborhood: str = "8",
                 context_radius: int = 1,
                 distance_mode: str = "uniform",
                 distance_gamma: float = 0.7,
                 store_device: t.Union[str, torch.device] = "cpu",
                 store_dtype: torch.dtype = torch.float16,
                 keep_rows_up: int = 4,
                 keep_rows_down: int = 4):
        super().__init__()
        assert 0.0 <= alpha <= 1.0
        assert 0.0 <= prefill_m <= 1.0
        assert neighborhood in ("4", "8")
        assert isinstance(context_radius, int) and context_radius >= 1
        assert distance_mode in ("uniform", "exp", "inverse")
        if distance_mode == "exp":
            assert 0.0 < distance_gamma <= 1.0

        self.in_ch = int(in_ch)
        self.r = int(r)
        self.alpha = float(alpha)
        self.prefill_m = float(prefill_m)
        self.include_center = bool(include_center)

        self.neighborhood = str(neighborhood)
        self.context_radius = int(context_radius)
        self.distance_mode = str(distance_mode)
        self.distance_gamma = float(distance_gamma)

        # CBAM CA trunk
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.in_ch, max(1, self.in_ch // self.r)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, self.in_ch // self.r), self.in_ch)
        )

        # Bank: [Y, X, C] + mask [Y, X]
        self.y_anchor_num: int = 0
        self.x_anchor_num: int = 0
        self.gain_bank: torch.Tensor | None = None   # [Y, X, C]
        self.init_mask: torch.Tensor | None = None   # [Y, X] bool

        # Neighbor cache: neigh_cache[y][x] = Tensor[K,3] on store_device (yy,xx,d)
        self.neigh_cache: list[list[torch.Tensor | None]] | None = None

        self.store_device = torch.device(store_device)
        self.store_dtype = store_dtype

        # eviction window (we don't really need it for CBAM, but keep for API)
        self.keep_rows_up = int(keep_rows_up)
        self.keep_rows_down = int(keep_rows_down)

    # ---------- grid / neighbor helpers ----------

    def _in_bounds(self, y: int, x: int) -> bool:
        return (0 <= y < self.y_anchor_num) and (0 <= x < self.x_anchor_num)

    def _neigh_coords_radius(self, y: int, x: int) -> list[tuple[int, int, int]]:
        R = self.context_radius
        out = []
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                yy, xx = y + dy, x + dx
                if dy == 0 and dx == 0:
                    continue
                if not self._in_bounds(yy, xx):
                    continue
                if self.neighborhood == "4":
                    d = abs(dy) + abs(dx)
                    if 1 <= d <= R:
                        out.append((yy, xx, d))
                else:  # "8"
                    d = max(abs(dy), abs(dx))
                    if 1 <= d <= R:
                        out.append((yy, xx, d))
        out.sort(key=lambda t: t[2])
        return out

    def _dist_weights_vec(self, d: torch.Tensor) -> torch.Tensor:
        if self.distance_mode == "uniform":
            return torch.ones_like(d)
        if self.distance_mode == "exp":
            return self.distance_gamma ** (d - 1.0)
        if self.distance_mode == "inverse":
            return 1.0 / d
        return torch.ones_like(d)

    @staticmethod
    def _first_idx(v) -> int:
        if v is None:
            return -1
        if isinstance(v, numbers.Integral):
            return int(v)
        if torch.is_tensor(v):
            if v.numel() == 0:
                return -1
            return int(v.view(-1)[0].item())
        if isinstance(v, (list, tuple)):
            if not v:
                return -1
            return int(v[0])
        return -1
    # ---------- bank lifecycle ----------

    def init_collection(self, y_anchor_num: int, x_anchor_num: int) -> None:
        self.y_anchor_num = int(y_anchor_num)
        self.x_anchor_num = int(x_anchor_num)

        self.gain_bank = torch.zeros(
            self.y_anchor_num,
            self.x_anchor_num,
            self.in_ch,
            device=self.store_device,
            dtype=self.store_dtype,
        )
        self.init_mask = torch.zeros(
            self.y_anchor_num,
            self.x_anchor_num,
            device=self.store_device,
            dtype=torch.bool,
        )

        # Precompute neighbor lists as small tensors
        self.neigh_cache = [[None for _ in range(self.x_anchor_num)]
                            for _ in range(self.y_anchor_num)]
        for y in range(self.y_anchor_num):
            for x in range(self.x_anchor_num):
                lst = self._neigh_coords_radius(y, x)
                if len(lst) == 0:
                    self.neigh_cache[y][x] = None
                else:
                    # Tensor[K,3] (yy,xx,d) on store_device
                    arr = torch.tensor(lst, device=self.store_device, dtype=torch.int16)
                    self.neigh_cache[y][x] = arr

    @torch.no_grad()
    def reset_bank(self) -> None:
        if self.init_mask is not None:
            self.init_mask.zero_()

    @torch.no_grad()
    def _bank_update(self, y: int, x: int, gain: Tensor) -> None:
        if self.gain_bank is None or self.init_mask is None:
            return
        if not self._in_bounds(y, x):
            return

        if gain.device != self.store_device or gain.dtype != self.store_dtype:
            g = gain.to(device=self.store_device, dtype=self.store_dtype)
        else:
            g = gain

        if not bool(self.init_mask[y, x]):
            self.gain_bank[y, x].copy_(g)
            self.init_mask[y, x] = True
        else:
            old = self.gain_bank[y, x]
            new = (1.0 - self.prefill_m) * old + self.prefill_m * g
            self.gain_bank[y, x].copy_(new)

    def _bank_get_vec(self, ys: torch.Tensor, xs: torch.Tensor,
                      like: Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized neighbor fetch:
          ys, xs: [K] indices (on store_device)
        Returns:
          gains: [K, C] (on like.device/like.dtype)
          valid: [K] bool mask (init_mask && in-bounds)
        """
        K = ys.numel()
        if self.gain_bank is None or self.init_mask is None or K == 0:
            return (torch.zeros(K, self.in_ch, device=like.device, dtype=like.dtype),
                    torch.zeros(K, device=like.device, dtype=torch.bool))

        ys = ys.long()
        xs = xs.long()
        mask_in = (0 <= ys) & (ys < self.y_anchor_num) & \
                  (0 <= xs) & (xs < self.x_anchor_num)
        if not mask_in.any():
            return (torch.zeros(K, self.in_ch, device=like.device, dtype=like.dtype),
                    torch.zeros(K, device=like.device, dtype=torch.bool))

        ys_v = ys[mask_in]
        xs_v = xs[mask_in]
        init = self.init_mask[ys_v, xs_v]  # [K_valid] bool on store_device
        if not init.any():
            return (torch.zeros(K, self.in_ch, device=like.device, dtype=like.dtype),
                    torch.zeros(K, device=like.device, dtype=torch.bool))

        ys_v2 = ys_v[init]
        xs_v2 = xs_v[init]

        gains_v = self.gain_bank[ys_v2, xs_v2]  # [K2, C] on store_device
        gains_all = torch.zeros(K, self.in_ch,
                                device=like.device,
                                dtype=like.dtype)
        valid_all = torch.zeros(K, device=like.device, dtype=torch.bool)

        # map back into full K indices
        idx_in_full = mask_in.nonzero(as_tuple=False).view(-1)[init]
        gains_all[idx_in_full] = gains_v.to(device=like.device, dtype=like.dtype)
        valid_all[idx_in_full] = True
        return gains_all, valid_all

    @torch.no_grad()
    def _evict_rows_around(self, center_x: int) -> None:
        # For CBAM banks, eviction is optional; we keep it no-op by default.
        if self.init_mask is None:
            return
        if center_x < 0 or self.x_anchor_num <= 0:
            return
        lo = max(0, center_x - self.keep_rows_up)
        hi = min(self.x_anchor_num - 1, center_x + self.keep_rows_down)
        # If you really want to evict, uncomment:
        # if lo > 0:
        #     self.init_mask[:, :lo] = False
        # if hi < self.x_anchor_num - 1:
        #     self.init_mask[:, hi+1:] = False

    # ---------- core CA ----------

    def _compute_gain(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.mlp(self.avg(x)) + self.mlp(self.max(x)))  # [B,C]

    def forward(self,
                x: Tensor,
                *,
                y_anchor=None, x_anchor=None,
                pre_y_anchor=None, pre_x_anchor=None,
                **_) -> Tensor:
        B, C, H, W = x.shape
        s_all = self._compute_gain(x)  # [B,C]

        if (not self.training) and B <= 2:
            s_use = s_all.clone()

            ya = self._first_idx(y_anchor)
            xa = self._first_idx(x_anchor)

            # write anchor to bank
            if ya != -1 and xa != -1:
                self._bank_update(ya, xa, s_all[0])

            # prefetch sample -> bank update
            if B == 2 and pre_y_anchor is not None and pre_x_anchor is not None:
                py = self._first_idx(pre_y_anchor)
                px = self._first_idx(pre_x_anchor)
                if py != -1 and px != -1:
                    self._bank_update(py, px, s_all[1])

            # anchor mixing
            if ya != -1 and xa != -1 and self.neigh_cache is not None:
                gains_list = []
                weight_list = []

                # center
                if self.include_center and self.gain_bank is not None:
                    ys_c = torch.tensor([ya], device=self.store_device)
                    xs_c = torch.tensor([xa], device=self.store_device)
                    g_c, init_c = self._bank_get_vec(ys_c, xs_c, like=x)
                    if init_c[0]:
                        gains_list.append(g_c[0])
                        weight_list.append(self._dist_weights_vec(
                            torch.tensor([0.0], device=x.device, dtype=x.dtype)
                        )[0])

                neigh = self.neigh_cache[ya][xa]
                if neigh is not None:
                    ys = neigh[:, 0].to(device=self.store_device)
                    xs = neigh[:, 1].to(device=self.store_device)
                    ds = neigh[:, 2].to(device=x.device, dtype=x.dtype).abs()

                    g_nb, init_nb = self._bank_get_vec(ys, xs, like=x)
                    if init_nb.any():
                        w_d = self._dist_weights_vec(ds)
                        w = w_d * init_nb.to(dtype=x.dtype)

                        valid_idx = init_nb.nonzero(as_tuple=False).view(-1)
                        if valid_idx.numel() > 0:
                            gains_list.append(g_nb[valid_idx])
                            weight_list.append(w[valid_idx])

                if len(gains_list) > 0:
                    G = torch.cat(
                        [g.unsqueeze(0) if g.ndim == 1 else g for g in gains_list],
                        dim=0
                    )  # [N,C]
                    w_all = torch.cat(
                        [w.view(-1) for w in weight_list],
                        dim=0
                    )  # [N]
                    w_all = w_all / (w_all.sum() + 1e-12)
                    g_avg = (G * w_all[:, None]).sum(dim=0)  # [C]
                    s_use[0] = (1.0 - self.alpha) * s_all[0] + self.alpha * g_avg
                else:
                    s_use[0] = s_all[0]
            else:
                s_use[0] = s_all[0]

            if xa != -1:
                self._evict_rows_around(xa)

            return x * s_use[:, :, None, None]

        s_use = s_all.clone()
        return x * s_use[:, :, None, None]


# ============================================================
# MarginBankConv (memory-bounded halo banks, on GPU)
# ============================================================

def scan_modules_for_raw_padding(net):
    bad_convs, bad_trans, bad_static = [], [], []

    for name, m in net.named_modules():
        if (isinstance(m, nn.Conv2d) and
                not isinstance(m, MarginBankConv) and
                max(m.padding) > 0):
            bad_convs.append((name, m.padding, m.padding_mode))

        if isinstance(m, (ReflectionPad2d, ReplicationPad2d, ConstantPad2d)):
            bad_static.append((name, type(m).__name__, m.padding))

    return bad_convs, bad_trans, bad_static


def assert_no_raw_padding(net):
    bad_convs, bad_trans, bad_static = scan_modules_for_raw_padding(net)

    msgs = []
    if bad_convs:
        msgs.append(f"raw Conv2d padding: {bad_convs}")
    print(f"WARNING: ConvTranspose2d padding leaks detected: {bad_trans}")
    if bad_static:
        msgs.append(f"static Pad2d layers: {bad_static}")

    if msgs:
        raise RuntimeError("Padding leaks detected →\n  " + "\n  ".join(msgs))


def _call_with_anchor(mod, x, anchor_kw):
    if isinstance(mod, (MarginBankConv, MarginBankPad2d)):
        return mod(x, **anchor_kw)

    if isinstance(mod, (nn.Sequential, nn.ModuleList)):
        for sub in mod:
            x = _call_with_anchor(sub, x, anchor_kw)
        return x

    try:
        return mod(x, **anchor_kw)
    except TypeError:
        return mod(x)


_call = _call_with_anchor  # alias


def init_prefetch_margin(model: nn.Module, y_anchor_num: int, x_anchor_num: int) -> None:
    """
    Initialize collection grids for all margin/attention banked layers.
    """
    for _, layer in model.named_modules():
        if isinstance(layer, (MarginBankConv, MarginBankPad2d, AnchorBankChannelAttention)):
            layer.init_collection(y_anchor_num=y_anchor_num, x_anchor_num=x_anchor_num)


class MarginBankConv(nn.Module):
    """
    Conv2d wrapper that caches margins per anchor and pastes neighbour halos.

    Memory-optimized version:
      - banks are [Y, X_bank, ...] instead of [Y, X_full, ...]
      - we keep only a horizontal window of columns around the scan head
        controlled by keep_rows_up / keep_rows_down
      - lookup is done via global2slot / slot2global mapping tensors
        on store_device (GPU when bank_device='cuda')
    """

    def __init__(
        self,
        original_conv,
        padding: int = 1,
        dim: int = 128,
        padding_mode: str = "replicate",
        device: t.Union[str, torch.device] = "cuda",
        store_device: t.Union[str, torch.device] = "cuda",
        store_dtype: torch.dtype = torch.float16,
        keep_rows_up: int = 2,
        keep_rows_down: int = 2,
    ):
        super().__init__()
        self.conv = original_conv
        self.padding = padding
        self.dim = dim
        self.device = torch.device(device)
        self.padding_mode = padding_mode

        # Anchor lattice size
        self.y_anchor_num = 0       # full Y
        self.x_anchor_num = 0       # full X
        self.x_bank = 0             # how many columns we actually store

        # Tensor banks for margins (allocated lazily once we know C,H,W)
        self.bank_top = None    # [Y, X_bank, C, p, W]
        self.bank_bottom = None
        self.bank_left = None   # [Y, X_bank, C, H, p]
        self.bank_right = None
        self.bank_tl = None     # [Y, X_bank, C, p, p]
        self.bank_tr = None
        self.bank_bl = None
        self.bank_br = None

        # Boolean mask: which anchors (y, slot) have valid margins
        self.init_mask: torch.Tensor | None = None  # [Y, X_bank] bool on store_device

        # For consistent shapes
        self._C = None
        self._H = None
        self._W = None

        self.store_device = torch.device(store_device)
        self.store_dtype = store_dtype

        self.keep_rows_up = int(keep_rows_up)
        self.keep_rows_down = int(keep_rows_down)

        # mapping: global x -> slot, and slot -> global x
        self.global2slot: torch.Tensor | None = None  # [X_full] int16, -1 if unused
        self.slot2global: torch.Tensor | None = None  # [X_bank] int16, -1 if unused

    # ---------- state dict compatibility ----------

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs
    ):
        legacy_w = prefix[:-1] + ".weight"
        legacy_b = prefix[:-1] + ".bias"
        if legacy_w in state_dict and prefix + "conv.weight" not in state_dict:
            state_dict[prefix + "conv.weight"] = state_dict.pop(legacy_w)
        if legacy_b in state_dict and prefix + "conv.bias" not in state_dict:
            state_dict[prefix + "conv.bias"] = state_dict.pop(legacy_b)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )

    # ---------- lattice init ----------

    def init_collection(self, y_anchor_num: int, x_anchor_num: int) -> None:
        """
        Called once per WSI to size the anchor lattice.
        Banks are allocated lazily once we know C,H,W.
        """
        self.y_anchor_num = int(y_anchor_num)
        self.x_anchor_num = int(x_anchor_num)

        window = self.keep_rows_up + self.keep_rows_down + 1
        self.x_bank = min(self.x_anchor_num, window)

        self.global2slot = torch.full(
            (self.x_anchor_num,),
            -1,
            device=self.store_device,
            dtype=torch.int16,
        )
        self.slot2global = torch.full(
            (self.x_bank,),
            -1,
            device=self.store_device,
            dtype=torch.int16,
        )

        self.init_mask = torch.zeros(
            self.y_anchor_num,
            self.x_bank,
            device=self.store_device,
            dtype=torch.bool,
        )

        # reset banks; shapes allocated lazily
        self.bank_top = self.bank_bottom = None
        self.bank_left = self.bank_right = None
        self.bank_tl = self.bank_tr = self.bank_bl = self.bank_br = None
        self._C = self._H = self._W = None

    def _in_bounds_global(self, y: int, x: int) -> bool:
        return (0 <= y < self.y_anchor_num) and (0 <= x < self.x_anchor_num)

    # ---------- lazy bank allocation ----------

    def _ensure_banks(self, C: int, H: int, W: int) -> None:
        """
        Allocate margin banks for this conv once we know C,H,W.
        """
        if self.y_anchor_num <= 0 or self.x_bank <= 0:
            return

        if (self.bank_top is not None and
                self._C == C and self._H == H and self._W == W):
            return  # already allocated

        self._C, self._H, self._W = int(C), int(H), int(W)
        Y, Xb = self.y_anchor_num, self.x_bank
        p = self.padding

        def alloc(*shape):
            return torch.zeros(
                *shape,
                device=self.store_device,
                dtype=self.store_dtype,
            )

        # stripes along width
        self.bank_top = alloc(Y, Xb, C, p, W)
        self.bank_bottom = alloc(Y, Xb, C, p, W)

        # stripes along height
        self.bank_left = alloc(Y, Xb, C, H, p)
        self.bank_right = alloc(Y, Xb, C, H, p)

        # corners p x p
        self.bank_tl = alloc(Y, Xb, C, p, p)
        self.bank_tr = alloc(Y, Xb, C, p, p)
        self.bank_bl = alloc(Y, Xb, C, p, p)
        self.bank_br = alloc(Y, Xb, C, p, p)

        if self.init_mask is None:
            self.init_mask = torch.zeros(
                Y, Xb, device=self.store_device, dtype=torch.bool
            )
        else:
            self.init_mask.zero_()

    # ---------- slot mapping ----------

    def _alloc_slot_for_x(self, x: int) -> int:
        """
        Map a global column index x (0..X_full-1) to a bank slot (0..X_bank-1).
        Reuse or evict slots as needed.
        """
        if self.global2slot is None or self.slot2global is None:
            raise RuntimeError("init_collection must be called before forward")

        x = int(x)
        if not (0 <= x < self.x_anchor_num):
            return -1

        slot = int(self.global2slot[x].item())
        if slot != -1:
            return slot

        # free slot?
        free = (self.slot2global == -1).nonzero(as_tuple=False)
        if free.numel() > 0:
            slot = int(free[0].item())
        else:
            # evict farthest in |global_x - x|
            g = self.slot2global.to(dtype=torch.float32)
            d = (g - float(x)).abs()
            slot = int(torch.argmax(d).item())

            old_x = int(self.slot2global[slot].item())
            if old_x != -1:
                self.global2slot[old_x] = -1
                if self.init_mask is not None:
                    self.init_mask[:, slot] = False

        self.slot2global[slot] = x
        self.global2slot[x] = slot

        return slot

    @torch.no_grad()
    def _evict_rows_around(self, center_x: int) -> None:
        """
        Mark anchors outside [center_x - keep_rows_up, center_x + keep_rows_down]
        as unused by clearing mappings & init_mask for those slots.
        """
        if (self.init_mask is None or
                self.slot2global is None or
                self.global2slot is None or
                center_x < 0 or
                self.x_anchor_num <= 0):
            return

        lo = max(0, center_x - self.keep_rows_up)
        hi = min(self.x_anchor_num - 1, center_x + self.keep_rows_down)

        g = self.slot2global.clone()  # [X_bank]
        mask = (g != -1) & ((g < lo) | (g > hi))
        evict_slots = mask.nonzero(as_tuple=False).view(-1)
        if evict_slots.numel() == 0:
            return

        old_xs = self.slot2global[evict_slots].to(dtype=torch.long)
        self.slot2global[evict_slots] = -1
        self.global2slot[old_xs] = -1
        self.init_mask[:, evict_slots] = False

    # ---------- margin write ----------

    @torch.no_grad()
    def _update_slot_from_input(self,
                                x_slice: Tensor,   # [1,C,H,W]
                                ya: int, xa: int,
                                inner: int) -> None:
        """
        Extract all 8 margins from x_slice and store them into the banks at (ya, slot(xa)).
        """
        if not self._in_bounds_global(ya, xa):
            return

        B, C, H, W = x_slice.shape
        assert B == 1, "MarginBankConv assumes per-tile B=1"

        self._ensure_banks(C, H, W)
        if self.bank_top is None:
            return

        slot = self._alloc_slot_for_x(xa)
        if slot == -1:
            return

        p = self.padding
        o = int(max(0, inner))
        o = min(o, H - p, W - p)

        top = x_slice[:, :, o:o+p, :]
        bottom = x_slice[:, :, H-o-p:H-o, :]
        left = x_slice[:, :, :, o:o+p]
        right = x_slice[:, :, :, W-o-p:W-o]

        tl = x_slice[:, :, o:o+p,               o:o+p]
        tr = x_slice[:, :, o:o+p,               W-o-p:W-o]
        bl = x_slice[:, :, H-o-p:H-o,           o:o+p]
        br = x_slice[:, :, H-o-p:H-o,           W-o-p:W-o]

        def to_bank(t: Tensor) -> Tensor:
            t = t.detach().to(device=self.store_device, dtype=self.store_dtype)
            return t.squeeze(0)

        self.bank_top[ya, slot].copy_(to_bank(top))
        self.bank_bottom[ya, slot].copy_(to_bank(bottom))
        self.bank_left[ya, slot].copy_(to_bank(left))
        self.bank_right[ya, slot].copy_(to_bank(right))
        self.bank_tl[ya, slot].copy_(to_bank(tl))
        self.bank_tr[ya, slot].copy_(to_bank(tr))
        self.bank_bl[ya, slot].copy_(to_bank(bl))
        self.bank_br[ya, slot].copy_(to_bank(br))

        if self.init_mask is not None:
            self.init_mask[ya, slot] = True

    # ---------- runtime conversion helper ----------

    def _to_runtime(self, t: Tensor, like: Tensor) -> Tensor:
        if t.device == like.device and t.dtype == like.dtype:
            return t
        return t.to(device=like.device, dtype=like.dtype, non_blocking=True)

    # ---------- margin read ----------

    def _fetch_margin(self,
                      bank: torch.Tensor | None,
                      y: int, x_global: int,
                      like: Tensor) -> Tensor | None:
        if (bank is None or
                self.init_mask is None or
                self.global2slot is None or
                self.slot2global is None):
            return None
        if not (0 <= y < self.y_anchor_num):
            return None
        if not (0 <= x_global < self.x_anchor_num):
            return None

        slot = int(self.global2slot[int(x_global)].item())
        if slot == -1:
            return None
        if not bool(self.init_mask[y, slot]):
            return None

        m = bank[y, slot]
        m_rt = self._to_runtime(m, like)
        return m_rt.unsqueeze(0)

    def _retrieve_margin(self,
                         ya: int, xa: int,
                         like: Tensor) -> dict[str, t.Optional[Tensor]]:
        if self.bank_top is None or self.init_mask is None:
            return dict(up=None, down=None, left=None, right=None,
                        tl=None, tr=None, bl=None, br=None)

        neigh = dict(
            up=self._fetch_margin(self.bank_bottom, ya - 1, xa, like),
            down=self._fetch_margin(self.bank_top, ya + 1, xa, like),
            left=self._fetch_margin(self.bank_right, ya, xa - 1, like),
            right=self._fetch_margin(self.bank_left, ya, xa + 1, like),
            tl=self._fetch_margin(self.bank_br, ya - 1, xa - 1, like),
            tr=self._fetch_margin(self.bank_bl, ya - 1, xa + 1, like),
            bl=self._fetch_margin(self.bank_tr, ya + 1, xa - 1, like),
            br=self._fetch_margin(self.bank_tl, ya + 1, xa + 1, like),
        )
        return neigh

    # ---------- forward ----------

    def forward(self, x: Tensor,
                y_anchor=None, x_anchor=None, pre_y_anchor=None, pre_x_anchor=None,
                inner: t.Union[int, list[int], None] = None,
                **kw) -> Tensor:
        B, C, H, W = x.shape
        p = self.padding

        def to_list(v, L):
            if v is None:
                return [-1] * L
            if isinstance(v, int):
                return [v] * L
            if len(v) != L:
                raise ValueError(f"Expected {L} anchors, got {len(v)}")
            return v

        y_anchor = to_list(y_anchor, 1)
        x_anchor = to_list(x_anchor, 1)
        pre_y_anchor = to_list(pre_y_anchor, max(B - 1, 0))
        pre_x_anchor = to_list(pre_x_anchor, max(B - 1, 0))

        anchor_y_all = y_anchor + pre_y_anchor
        anchor_x_all = x_anchor + pre_x_anchor

        def to_inner_list(v, L):
            if v is None:
                return [0] * L
            if isinstance(v, int):
                return [v] * L
            if len(v) == 1:
                return v * L
            if len(v) != L:
                raise ValueError(f"Expected {L} 'inner' entries, got {len(v)}")
            return v

        inner_all = to_inner_list(inner, B)

        for i in range(B):
            ya, xa = anchor_y_all[i], anchor_x_all[i]
            if ya != -1 and xa != -1:
                self._update_slot_from_input(
                    x[i:i+1],
                    ya, xa,
                    inner_all[i]
                )

        # pad input
        if self.padding_mode == "constant":
            x_padded = F.pad(x, (p, p, p, p), mode="constant", value=0.0)
        else:
            x_padded = F.pad(x, (p, p, p, p), mode=self.padding_mode)

        # halo pasting for head sample only
        B_iter = 1
        for i in range(B_iter):
            ya, xa = anchor_y_all[i], anchor_x_all[i]
            if ya == -1 or xa == -1:
                continue

            xi = x_padded[i:i+1]
            row_mid = slice(p, p + H)
            col_mid = slice(p, p + W)

            row_top = slice(row_mid.start, row_mid.start + p)
            row_bottom = slice(row_mid.stop - p, row_mid.stop)
            col_left = slice(col_mid.start, col_mid.start + p)
            col_right = slice(col_mid.stop - p, col_mid.stop)

            neigh = self._retrieve_margin(ya, xa, like=xi)

            def paste_(dst, src):
                if src is None:
                    return
                dst.copy_(src)

            if neigh["up"] is None:
                neigh["up"] = torch.flip(xi[:, :, row_top, col_mid], (2,))
            if neigh["down"] is None:
                neigh["down"] = torch.flip(xi[:, :, row_bottom, col_mid], (2,))
            if neigh["left"] is None:
                neigh["left"] = torch.flip(xi[:, :, row_mid, col_left], (3,))
            if neigh["right"] is None:
                neigh["right"] = torch.flip(xi[:, :, row_mid, col_right], (3,))

            if neigh["tl"] is None:
                neigh["tl"] = torch.flip(xi[:, :, row_top, col_left], (2, 3))
            if neigh["tr"] is None:
                neigh["tr"] = torch.flip(xi[:, :, row_top, col_right], (2, 3))
            if neigh["bl"] is None:
                neigh["bl"] = torch.flip(xi[:, :, row_bottom, col_left], (2, 3))
            if neigh["br"] is None:
                neigh["br"] = torch.flip(xi[:, :, row_bottom, col_right], (2, 3))

            paste_(xi[:, :, :p, col_mid], neigh["up"])
            paste_(xi[:, :, -p:, col_mid], neigh["down"])
            paste_(xi[:, :, row_mid, :p], neigh["left"])
            paste_(xi[:, :, row_mid, -p:], neigh["right"])

            paste_(xi[:, :, :p, :p], neigh["tl"])
            paste_(xi[:, :, :p, -p:], neigh["tr"])
            paste_(xi[:, :, -p:, :p], neigh["bl"])
            paste_(xi[:, :, -p:, -p:], neigh["br"])

            x_padded[i:i+1] = xi

        head_x = x_anchor[0] if x_anchor else -1
        if head_x != -1:
            self._evict_rows_around(head_x)

        return self.conv(x_padded)


def _clone_conv_no_pad(conv: nn.Conv2d) -> nn.Conv2d:
    clone = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=0,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode='zeros',
    )
    clone.weight.data.copy_(conv.weight.data)
    if conv.bias is not None:
        clone.bias.data.copy_(conv.bias.data)
    return clone


def wrap_conv_with_marginbank(
    module: nn.Module,
    *,
    padding_mode: str = "replicate",
    bank_device: t.Union[str, torch.device] = "cuda",
    bank_dtype: torch.dtype = torch.float16,
):
    """
    Recursively wrap Conv2d layers that have padding with MarginBankConv,
    storing halo banks on 'bank_device' with dtype 'bank_dtype'.
    """
    for name, sub in list(module.named_children()):
        if isinstance(sub, nn.Conv2d) and max(sub.padding) > 0:
            new = _clone_conv_no_pad(sub)
            setattr(
                module,
                name,
                MarginBankConv(
                    original_conv=new,
                    padding=sub.padding[0],
                    dim=new.out_channels,
                    padding_mode=padding_mode,
                    store_device=bank_device,
                    store_dtype=bank_dtype,
                )
            )
            continue
        wrap_conv_with_marginbank(
            sub,
            padding_mode=padding_mode,
            bank_device=bank_device,
            bank_dtype=bank_dtype,
        )


class _Identity(nn.Module):
    def forward(self, x):
        return x


class MarginBankPad2d(nn.Module):
    """
    Halo-aware substitute for nn.Reflection/Replication/ConstantPad2d.
    """
    _MODE2BANK = dict(
        replicate="replicate",
        reflection="reflect",
        reflect="reflect",
        constant="constant",
    )

    def __init__(self,
                 padding: int | tuple[int, int, int, int] = 1,
                 mode: str = "replicate",
                 constant_value: float = 0.0):
        super().__init__()

        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)

        l, r, t, b = padding
        if not (l == r == t == b):
            raise ValueError("MarginBankPad2d supports only symmetric padding")
        self.p = l

        self.mode = mode
        self.constant_value = constant_value

        self.bank = MarginBankConv(
            original_conv=_Identity(),
            padding=self.p,
            dim=1,
            padding_mode=self._MODE2BANK[mode],
        )

    init_collection = lambda self, *a, **kw: self.bank.init_collection(*a, **kw)

    def forward(self, x: Tensor, **anchor_kw) -> Tensor:
        out = self.bank.forward(x, **anchor_kw)
        return out


def replace_static_pad(module: nn.Module):
    for name, child in list(module.named_children()):
        if isinstance(child, (ReflectionPad2d, ReplicationPad2d, ConstantPad2d)):
            pad_size = child.padding
            mode = ("reflect" if isinstance(child, ReflectionPad2d)
                    else "replicate" if isinstance(child, ReplicationPad2d)
                    else "constant")
            value = getattr(child, 'value', 0.0)
            setattr(module, name,
                    MarginBankPad2d(padding=pad_size, mode=mode,
                                    constant_value=value))
        else:
            replace_static_pad(child)


# ============================================================
# ResNet backbone, CBAM, U-Net-style decoder
# ============================================================

BOTTLENECK_DIM = 2048  # ResNet50 bottleneck dimension


def _conv3x3(in_c, out_c, stride=1, groups=1, padding=1):
    return nn.Conv2d(in_c, out_c, 3, stride, padding, groups=groups, bias=False)


def _conv1x1(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 1, stride, bias=False)


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d):
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = _conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = _conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = _conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, **anchor_kw):
        identity = x
        out = _call_with_anchor(self.conv1, x, anchor_kw)
        out = _call_with_anchor(self.bn1, out, anchor_kw)
        out = _call(self.relu, out, {})

        out = _call_with_anchor(self.conv2, out, anchor_kw)
        out = _call_with_anchor(self.bn2, out, anchor_kw)
        out = _call(self.relu, out, {})

        out = _call_with_anchor(self.conv3, out, anchor_kw)
        out = _call_with_anchor(self.bn3, out, anchor_kw)

        if self.downsample is not None:
            identity = _call_with_anchor(self.downsample, x, anchor_kw)

        return self.relu(out + identity)


class ResNetBackbone(nn.Module):
    """ResNet-50 feature extractor (conv→layer4)."""

    def __init__(self, layers: List[int] = [3, 4, 6, 3],
                 in_ch: int = 10,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d):
        super().__init__()
        self.inplanes, self.dilation = 64, 1
        self.groups, self.base_width = 1, 64

        self.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(_Bottleneck, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(_Bottleneck, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(_Bottleneck, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(_Bottleneck, 512, layers[3], stride=2, norm_layer=norm_layer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
                if getattr(m, "weight", None) is not None:
                    nn.init.constant_(m.weight, 1)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, _Bottleneck):
                w = getattr(m.bn3, "weight", None)
                if w is not None:
                    nn.init.constant_(w, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample,
                        self.groups, self.base_width, self.dilation, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_ch, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, in_ch // r),
            nn.ReLU(),
            nn.Linear(in_ch // r, in_ch),
        )

    def forward(self, x, **anchor_kw):
        s = torch.sigmoid(self.mlp(self.avg(x)) + self.mlp(self.max(x)))
        return x * s.unsqueeze(2).unsqueeze(3)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, 1, 3)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x, **anchor_kw):
        s = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        s = _call_with_anchor(self.conv, s, anchor_kw)
        s = self.bn(s)
        s = torch.sigmoid(s)
        return x * s


class CBAM(nn.Module):
    def __init__(self, in_ch, r=16):
        super().__init__()
        self.ca = ChannelAttention(in_ch, r)
        self.sa = SpatialAttention()

    def forward(self, x, **anchor_kw):
        return _call_with_anchor(self.sa, _call_with_anchor(self.ca, x, anchor_kw), anchor_kw)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer=nn.BatchNorm2d):
        super().__init__()
        act = nn.ReLU(True)
        self.pad = nn.ReflectionPad2d(1)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
            norm_layer(out_ch), act)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 3, 1, 1), norm_layer(out_ch), act,
            nn.Conv2d(out_ch, out_ch, 3, 1, 1), norm_layer(out_ch), act)

    def forward(self, x1, x2, anchor_kw=None):
        anchor_kw = anchor_kw or {}
        up_feat = _call_with_anchor(self.up, x1, anchor_kw)
        out = _call_with_anchor(self.conv, torch.cat([x2, up_feat], 1),
                                anchor_kw)
        return out


# ============================================================
# AttUnetGenerator with halo banks + CBAM bank
# ============================================================

class AttUnetGenerator(BaseNetwork):
    def __init__(self,
                 opt,
                 norm_layer=nn.BatchNorm2d,
                 fpn_feature='decoder',
                 num_channels=10,
                 use_halo_banks: bool = True,
                 use_cbam_bank: bool = True,
                 bank_device: str = "cuda",
                 cbam_radius: int = 2):
        super().__init__()
        self.num_channels = num_channels
        self.use_halo_banks = bool(use_halo_banks)
        self.use_cbam_bank = bool(use_cbam_bank)
        self.bank_device = str(bank_device)
        self.cbam_radius = int(cbam_radius)

        backbone = ResNetBackbone(in_ch=self.num_channels, norm_layer=norm_layer)
        self.encoder = nn.Module()
        self.encoder.conv1 = backbone.conv1
        self.encoder.bn1 = backbone.bn1
        self.encoder.relu = backbone.relu
        self.encoder.maxpool = backbone.maxpool
        self.encoder.layer1 = backbone.layer1
        self.encoder.layer2 = backbone.layer2
        self.encoder.layer3 = backbone.layer3
        self.encoder.layer4 = backbone.layer4

        self.fpn_feature = fpn_feature
        self.ngf = 64
        self.n_down = 5
        self.out_nc = 3
        for i in range(self.n_down - 1):
            mult = 2 ** (self.n_down - i)
            setattr(self, f'up{i}', Up(self.ngf * mult,
                                       self.ngf * mult // 2 if i != self.n_down - 2 else self.ngf,
                                       norm_layer=norm_layer))
            setattr(self, f'cbam{i}', CBAM(self.ngf * mult))
        setattr(self, f'cbam{self.n_down - 1}', CBAM(self.ngf))

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(self.ngf, self.ngf // 2, 4, 2, 1),
            norm_layer(self.ngf // 2), nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(self.ngf // 2, self.out_nc, 7, 1, 0),
            nn.Tanh()
        )

        if self.use_halo_banks or self.use_cbam_bank:
            eff_bank_device = self.bank_device
            if eff_bank_device == "cuda" and not torch.cuda.is_available():
                print("[AttUnetGenerator] bank_device='cuda' requested but CUDA not "
                      "available; using CPU banks instead.")
                eff_bank_device = "cpu"

            if eff_bank_device == "cuda" and hasattr(torch, "float8_e4m3fn"):
                bank_dtype = torch.float8_e4m3fn  # type: ignore[attr-defined]
            else:
                bank_dtype = torch.float16

            if self.use_halo_banks:
                wrap_conv_with_marginbank(
                    self,
                    padding_mode='replicate',
                    bank_device=eff_bank_device,
                    bank_dtype=bank_dtype,
                )
                replace_static_pad(self)

            if self.use_cbam_bank:
                swap_cbam_channel_attention_with_banked(
                    self,
                    alpha=1.0,
                    prefill_m=1.0,
                    include_center=True,
                    neighborhood="8",
                    context_radius=self.cbam_radius,
                    distance_mode="uniform",
                    distance_gamma=0.7,
                    store_device=eff_bank_device,
                    store_dtype=torch.float16,
                )

    @torch.no_grad()
    def forward_prefetch(self, x,
                         *,
                         y_anchor=None, x_anchor=None,
                         pre_y_anchor=None, pre_x_anchor=None,
                         layers=(0, 1, 2, 3, 4),
                         return_feats: bool = False,
                         overlap_px: int = 0):
        if overlap_px > 0:
            inner_1 = overlap_px // 1
            inner_2 = overlap_px // 2
            inner_4 = overlap_px // 4
            inner_8 = overlap_px // 8
            inner_16 = overlap_px // 16
            inner_32 = overlap_px // 32
        else:
            inner_1 = inner_2 = inner_4 = inner_8 = inner_16 = inner_32 = 0

        kw1 = dict(
            y_anchor=y_anchor, x_anchor=x_anchor,
            pre_y_anchor=pre_y_anchor, pre_x_anchor=pre_x_anchor,
            inner=inner_1
        )
        kw2 = dict(
            y_anchor=y_anchor, x_anchor=x_anchor,
            pre_y_anchor=pre_y_anchor, pre_x_anchor=pre_x_anchor,
            inner=inner_2
        )
        kw4 = dict(
            y_anchor=y_anchor, x_anchor=x_anchor,
            pre_y_anchor=pre_y_anchor, pre_x_anchor=pre_x_anchor,
            inner=inner_4
        )
        kw8 = dict(
            y_anchor=y_anchor, x_anchor=x_anchor,
            pre_y_anchor=pre_y_anchor, pre_x_anchor=pre_x_anchor,
            inner=inner_8
        )
        kw16 = dict(
            y_anchor=y_anchor, x_anchor=x_anchor,
            pre_y_anchor=pre_y_anchor, pre_x_anchor=pre_x_anchor,
            inner=inner_16
        )
        kw32 = dict(
            y_anchor=y_anchor, x_anchor=x_anchor,
            pre_y_anchor=pre_y_anchor, pre_x_anchor=pre_x_anchor,
            inner=inner_32
        )

        # encoder
        f0 = _call_with_anchor(self.encoder.conv1, x, kw1)
        f = _call_with_anchor(self.encoder.bn1, f0, kw1)
        f = self.encoder.relu(f)
        f = self.encoder.maxpool(f)

        f1 = f
        for sub in self.encoder.layer1:
            f1 = sub(f1, **kw4)

        f2 = f1
        for sub in self.encoder.layer2:
            f2 = sub(f2, **kw8)

        f3 = f2
        for sub in self.encoder.layer3:
            f3 = sub(f3, **kw16)

        f4 = f3
        for sub in self.encoder.layer4:
            f4 = sub(f4, **kw32)

        if return_feats:
            feats = [f0, f1, f2, f3, f4]
            return feats

        deep = f4

        # decoder
        x_dec = _call_with_anchor(self.cbam0, deep, kw32)

        x_dec = self.up0(
            x_dec,
            _call_with_anchor(self.cbam1, f3, kw16),
            anchor_kw=kw16
        )

        x_dec = self.up1(
            x_dec,
            _call_with_anchor(self.cbam2, f2, kw8),
            anchor_kw=kw8
        )

        x_dec = self.up2(
            x_dec,
            _call_with_anchor(self.cbam3, f1, kw4),
            anchor_kw=kw4
        )

        x_dec = self.up3(
            x_dec,
            _call_with_anchor(self.cbam4, f0, kw2),
            anchor_kw=kw2
        )

        out_img = _call_with_anchor(self.final_up, x_dec, kw1)
        return out_img

    def forward(self, x, *, return_feats=False,
                encode_only=False, layers=(0, 1, 2, 3, 4), patch_ids=None, token_only=False):

        feats = []

        f0 = self.encoder.conv1(x)
        if 0 in layers:
            feats.append(f0)

        f = self.encoder.bn1(f0)
        f = self.encoder.relu(f)
        f = self.encoder.maxpool(f)

        for idx, block in enumerate((self.encoder.layer1,
                                     self.encoder.layer2,
                                     self.encoder.layer3,
                                     self.encoder.layer4), start=1):
            f = block(f)
            if idx in layers:
                feats.append(f)

        if encode_only:
            return [ft for idx, ft in enumerate(feats) if idx in layers]

        if token_only:
            return f

        deep = f
        feats_full = [deep] + feats[::-1][1:]
        x_dec = getattr(self, 'cbam0')(feats_full[0])
        for i in range(self.n_down - 1):
            x_dec = getattr(self, f'up{i}')(x_dec,
                                            getattr(self, f'cbam{i+1}')(feats_full[i+1]))

        out_img = self.final_up(x_dec)
        return (out_img, feats) if return_feats else out_img


# ============================================================
# LocalGlobalAttGenerator + CBAM swap
# ============================================================

def swap_cbam_channel_attention_with_banked(root: nn.Module,
                                            *,
                                            alpha: float = 0.10,
                                            prefill_m: float = 1.0,
                                            include_center: bool = True,
                                            neighborhood: str = "8",
                                            context_radius: int = 1,
                                            distance_mode: str = "uniform",
                                            distance_gamma: float = 0.7,
                                            store_device: t.Union[str, torch.device] = "cpu",
                                            store_dtype: torch.dtype = torch.float16) -> None:
    for m in root.modules():
        if m.__class__.__name__ == "CBAM" and hasattr(m, "ca") and hasattr(m.ca, "mlp"):
            in_ch = m.ca.mlp[-1].out_features
            hidden = m.ca.mlp[1].out_features
            r = max(1, in_ch // hidden)
            banked = AnchorBankChannelAttention(
                in_ch, r,
                alpha=alpha,
                prefill_m=prefill_m,
                include_center=include_center,
                neighborhood=neighborhood,
                context_radius=context_radius,
                distance_mode=distance_mode,
                distance_gamma=distance_gamma,
                store_device=store_device,
                store_dtype=store_dtype,
            )
            m.ca = banked  # swap in-place


class LocalGlobalAttGenerator(BaseNetwork):
    def __init__(self, opt=None, 
                 num_ch = 5,
                 norm_layer=nn.BatchNorm2d, fpn_feature='decoder',
                 ckpt_path='...',
                 bank_device: str = "cuda",
                 cbam_radius: int = 2,
                 anchor_enabled: bool = True):
        super().__init__()
        self.enc = AttUnetGenerator(
            opt,
            norm_layer,
            fpn_feature,
            num_channels=num_ch,
            use_halo_banks=anchor_enabled,
            use_cbam_bank=anchor_enabled,
            bank_device=bank_device,
            cbam_radius=cbam_radius,
        )

        self.b1, self.b2 = float(1.3), float(1.4)
        self.s0, self.s1, self.s2 = 0.9, 0.8, 0.2

        ckpt = torch.load(
            ckpt_path,
            map_location='cpu'
        )

        missing, unexpected = self.enc.load_state_dict(ckpt, strict=False)
        if len(missing) > 0:
            print(f"WARNING: missing keys when loading pretrained weights: {missing}")
        if len(unexpected) > 0:
            print(f"WARNING: unexpected keys when loading pretrained weights: {unexpected}")

        print(f"Loaded pretrained weights into LocalGlobalAttGenerator.")

        self.enc.encoder.eval()

    def forward(self, x):
        with torch.no_grad():
            f0 = self.enc.encoder.conv1(x)
            mf0 = f0
            f = self.enc.encoder.bn1(f0)
            f = self.enc.encoder.relu(f)
            f = self.enc.encoder.maxpool(f)

            feats = [mf0]
            for i in ['1', '2', '3', '4']:
                layer = getattr(self.enc.encoder, f'layer{i}')
                f = layer(f)
                feats.append(f)

            deep = feats[-1]

        deep = deep.detach()

        feat_stack = [deep] + feats[::-1][1:]
        x_dec = getattr(self.enc, 'cbam0')(feat_stack[0])
        for i in range(self.enc.n_down - 1):
            up = getattr(self.enc, f'up{i}')
            cb = getattr(self.enc, f'cbam{i + 1}')
            x_dec = up(x_dec, cb(feat_stack[i + 1]))

        x_dec = self.enc.final_up(x_dec)
        return x_dec
