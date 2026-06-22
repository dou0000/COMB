# COMB — Seamless Whole Slide Label-Free Virtual Staining

Official implementation of **"Seamless Whole Slide Label-Free Virtual Staining"** (MICCAI 2026).

We introduce the **Consistency Memory Bank (COMB)**, a label-free virtual staining framework that decouples context storage from computation to enforce spatial and channel consistency across patches without memory bottlenecks. COMB combines retrieval-based **local padding** for spatial continuity and **neighbor-aware CBAM** for channel-statistic stability, with a sliding-window memory schedule for tractable inference on gigapixel WSIs.

## Authors

Dou Hoon Kwark, Kianoush Falahkheirkhah, Ji-Hun Oh, Shirui Luo, Volodymyr Kindratenko\*, Rohit Bhargava\*

\*Contributed equally


## Training

```bash
python train.py --batch_size 8 --exp_name my_run --checkpoints_dir ./checkpoints
```

## Inference

```bash
python WSI_inference.py \
    --ckpt_path /path/to/checkpoint_net_G.pth \
    --input /path/to/roi.npy \
    --output ./out.png
```

Pass a directory of ROIs to batch-process:

```bash
python WSI_inference.py \
    --ckpt_path /path/to/checkpoint_net_G.pth \
    --input  /path/to/rois/ \
    --output /path/to/outputs/
```

### CBAM mixing

Defaults match the paper's Eq. 2 (α = 0.3, R = 2, center excluded). Pass `--alpha #` to override.

### Resolution scaling

For modalities at lower resolution than H&E (e.g. 2 µm/px → 0.5 µm/px), pass `--scaling_factor 0.25`.

### Normalization

Pass `--clip_max N` to clip the spectral input to `[0, N]` before normalizing.

## Citation

```bibtex
@inproceedings{comb_miccai2026,
  title  = {Seamless Whole Slide Label-Free Virtual Staining},
  author = {Kwark, Dou Hoon and Falahkheirkhah, Kianoush and Oh, Ji-Hun and Luo, Shirui and Kindratenko, Volodymyr and Bhargava, Rohit},
  booktitle = {MICCAI},
  year   = {2026}
}
```

## Acknowledgments

This codebase builds on Andonian et al.'s [Contrastive Feature Loss](https://github.com/alexandonian/contrastive-feature-loss).

## License

See `LICENSE`.
