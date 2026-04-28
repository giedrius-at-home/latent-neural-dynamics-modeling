# FB Ablation - 4-condition comparison

**Generated:** 2026-04-22 06:31

## Conditions

| key | flag | description |
|---|---|---|
| `rts_light` | suffix `rts_noclip` | max_eig=0.99999  fb=False |
| `fb_light` | suffix `fb_noclip` | max_eig=0.99999  fb=True |
| `rts_clip` | suffix `optA_rts_clip` | max_eig=0.9999   fb=False (via Option A) |
| `fb_clip` | suffix `(bare)` | max_eig=0.9999   fb=True |

## Test-split reconstruction + 1 s forecast (Pearson r)

| cell-mode | cond | Y recon r | Y recon RMSE | Z recon r | Z recon RMSE | Y fcst r | Z fcst r |
|---|---|---|---|---|---|---|---|
| PDI1_S2 behavioral | rts_light | 0.7435 | 0.623 | 0.0415 | 3192.9 | 0.0638 | 0.0041 |
| PDI1_S2 behavioral | fb_light | 0.7435 | 0.623 | 0.0134 | 3223.2 | 0.0580 | 0.0029 |
| PDI1_S2 behavioral | rts_clip | 0.7435 | 0.623 | 0.0415 | 3192.9 | 0.0638 | 0.0044 |
| PDI1_S2 behavioral | fb_clip | 0.7435 | 0.623 | 0.0142 | 3224.9 | 0.0580 | 0.0032 |
| PDI1_S4 behavioral | rts_light | 0.7090 | 2.134 | 0.0241 | 5448.9 | 0.0314 | -0.0002 |
| PDI1_S4 behavioral | fb_light | 0.7090 | 2.134 | 0.0075 | 5453.2 | 0.0291 | 0.0031 |
| PDI1_S4 behavioral | rts_clip | 0.7089 | 2.134 | 0.0241 | 5448.9 | 0.0314 | -0.0002 |
| PDI1_S4 behavioral | fb_clip | 0.7089 | 2.134 | 0.0075 | 5453.2 | 0.0291 | 0.0031 |
| PDI4_S2 behavioral | rts_light | 0.8879 | 0.294 | 0.0001 | 12911.9 | 0.1069 | -0.0562 |
| PDI4_S2 behavioral | fb_light | 0.8879 | 0.294 | -0.0031 | 12918.7 | 0.0997 | -0.0485 |
| PDI4_S2 behavioral | rts_clip | 0.8879 | 0.294 | -0.0000 | 12912.1 | 0.1069 | -0.0545 |
| PDI4_S2 behavioral | fb_clip | 0.8879 | 0.294 | -0.0040 | 12920.3 | 0.0997 | -0.0471 |
| PDI4_S3 behavioral | rts_light | 0.8126 | 0.699 | 0.0251 | 20959.1 | 0.0840 | -0.0215 |
| PDI4_S3 behavioral | fb_light | 0.8126 | 0.699 | 0.0181 | 21618.5 | 0.0745 | -0.0346 |
| PDI4_S3 behavioral | rts_clip | 0.8126 | 0.699 | 0.0248 | 20959.1 | 0.0841 | -0.0224 |
| PDI4_S3 behavioral | fb_clip | 0.8126 | 0.699 | 0.0063 | 21500.1 | 0.0746 | -0.0358 |
| PDI1_S2 laplacian | rts_light | 0.8607 | 0.662 | -0.0096 | 3.2 | 0.1075 | 0.0128 |
| PDI1_S2 laplacian | fb_light | 0.8607 | 0.662 | -0.0041 | 3.2 | 0.0995 | -0.0036 |
| PDI1_S2 laplacian | rts_clip | 0.8607 | 0.662 | -0.0096 | 3.2 | 0.1075 | 0.0128 |
| PDI1_S2 laplacian | fb_clip | 0.8607 | 0.662 | -0.0041 | 3.2 | 0.0995 | -0.0036 |
| PDI1_S4 laplacian | rts_light | 0.8043 | 2.282 | 0.0021 | 18.9 | 0.0309 | 0.0235 |
| PDI1_S4 laplacian | fb_light | 0.8043 | 2.282 | -0.0007 | 18.9 | 0.0255 | 0.0262 |
| PDI1_S4 laplacian | rts_clip | 0.8042 | 2.282 | 0.0021 | 18.9 | 0.0309 | 0.0235 |
| PDI1_S4 laplacian | fb_clip | 0.8042 | 2.282 | -0.0007 | 18.9 | 0.0255 | 0.0262 |
| PDI4_S2 laplacian | rts_light | 0.8447 | 0.361 | 0.0048 | 15.2 | 0.0503 | -0.0001 |
| PDI4_S2 laplacian | fb_light | 0.8447 | 0.361 | 0.0084 | 15.2 | 0.0425 | -0.0023 |
| PDI4_S2 laplacian | rts_clip | 0.8447 | 0.361 | 0.0048 | 15.2 | 0.0503 | -0.0001 |
| PDI4_S2 laplacian | fb_clip | 0.8447 | 0.361 | 0.0084 | 15.2 | 0.0425 | -0.0024 |
| PDI4_S3 laplacian | rts_light | 0.8713 | 0.396 | 0.0256 | 4.7 | 0.0985 | 0.0079 |
| PDI4_S3 laplacian | fb_light | 0.8713 | 0.396 | -0.0059 | 4.8 | 0.0886 | 0.0204 |
| PDI4_S3 laplacian | rts_clip | 0.8713 | 0.396 | 0.0256 | 4.7 | 0.0985 | 0.0079 |
| PDI4_S3 laplacian | fb_clip | 0.8713 | 0.396 | -0.0059 | 4.8 | 0.0886 | 0.0204 |

## Classification test BA (prediction mode)

| cell-mode | rts_light | fb_light | rts_clip | fb_clip |
|---|---|---|---|---|
| PDI1_S2 behavioral | 0.5420 | 0.5420 | 0.5410 | 0.5410 |
| PDI1_S4 behavioral | 0.7906 | 0.7906 | 0.7906 | 0.7906 |
| PDI4_S2 behavioral | 0.6835 | 0.6835 | 0.6835 | 0.6835 |
| PDI4_S3 behavioral | 0.7266 | 0.7266 | 0.7266 | 0.7266 |
| PDI1_S2 laplacian | 0.5868 | 0.5868 | 0.5868 | 0.5868 |
| PDI1_S4 laplacian | 0.7919 | 0.7919 | 0.7919 | 0.7919 |
| PDI4_S2 laplacian | 0.6181 | 0.6181 | 0.6181 | 0.6181 |
| PDI4_S3 laplacian | 0.8294 | 0.8294 | 0.8294 | 0.8294 |

## Classification test BA (forecast h=1.0 s, m=0.5 s)

| cell-mode | rts_light | fb_light | rts_clip | fb_clip |
|---|---|---|---|---|
| PDI1_S2 behavioral | 0.6801 | 0.6473 | 0.6801 | 0.6473 |
| PDI1_S4 behavioral | 0.7833 | 0.7833 | 0.7833 | 0.7833 |
| PDI4_S2 behavioral | 0.5491 | 0.5648 | 0.5491 | 0.5648 |
| PDI4_S3 behavioral | 0.6410 | 0.6410 | 0.6410 | 0.6553 |
| PDI1_S2 laplacian | 0.6176 | 0.6443 | 0.6176 | 0.6443 |
| PDI1_S4 laplacian | 0.7833 | 0.7694 | 0.7833 | 0.7694 |
| PDI4_S2 laplacian | 0.6619 | 0.7004 | 0.6619 | 0.7004 |
| PDI4_S3 laplacian | 0.5586 | 0.5015 | 0.5586 | 0.5015 |

## Classification test BA (forecast h=1.0 s, m=1.0 s)

| cell-mode | rts_light | fb_light | rts_clip | fb_clip |
|---|---|---|---|---|
| PDI1_S2 behavioral | 0.6882 | 0.6838 | 0.6882 | 0.6897 |
| PDI1_S4 behavioral | 0.7833 | 0.7833 | 0.7833 | 0.7833 |
| PDI4_S2 behavioral | 0.5641 | 0.5976 | 0.5641 | 0.5976 |
| PDI4_S3 behavioral | 0.6504 | 0.6498 | 0.6504 | 0.6564 |
| PDI1_S2 laplacian | 0.5580 | 0.5186 | 0.5580 | 0.5186 |
| PDI1_S4 laplacian | 0.7833 | 0.7694 | 0.7833 | 0.7694 |
| PDI4_S2 laplacian | 0.6668 | 0.6484 | 0.6668 | 0.6484 |
| PDI4_S3 laplacian | 0.6398 | 0.5701 | 0.6398 | 0.5701 |

## Classification test BA (forecast h=2.0 s, m=0.5 s)

| cell-mode | rts_light | fb_light | rts_clip | fb_clip |
|---|---|---|---|---|
| PDI1_S2 behavioral | 0.6086 | 0.5982 | 0.6086 | 0.5982 |
| PDI1_S4 behavioral | 0.7833 | 0.7833 | 0.7833 | 0.7833 |
| PDI4_S2 behavioral | 0.6832 | 0.6604 | 0.6832 | 0.6604 |
| PDI4_S3 behavioral | 0.6695 | 0.6850 | 0.6695 | 0.6850 |
| PDI1_S2 laplacian | 0.7426 | 0.7649 | 0.7426 | 0.7649 |
| PDI1_S4 laplacian | 0.7833 | 0.7833 | 0.7833 | 0.7833 |
| PDI4_S2 laplacian | 0.6648 | 0.6491 | 0.6648 | 0.6491 |
| PDI4_S3 laplacian | 0.7541 | 0.6981 | 0.7541 | 0.6981 |

## Classification test BA (forecast h=2.0 s, m=1.0 s)

| cell-mode | rts_light | fb_light | rts_clip | fb_clip |
|---|---|---|---|---|
| PDI1_S2 behavioral | 0.5893 | 0.6295 | 0.5893 | 0.6295 |
| PDI1_S4 behavioral | 0.7833 | 0.7833 | 0.7833 | 0.7833 |
| PDI4_S2 behavioral | 0.5291 | 0.6205 | 0.5291 | 0.6269 |
| PDI4_S3 behavioral | 0.5867 | 0.5718 | 0.5867 | 0.5718 |
| PDI1_S2 laplacian | 0.5298 | 0.5186 | 0.5298 | 0.5186 |
| PDI1_S4 laplacian | 0.7833 | 0.7833 | 0.7833 | 0.7833 |
| PDI4_S2 laplacian | 0.5890 | 0.6033 | 0.5890 | 0.6033 |
| PDI4_S3 laplacian | 0.6635 | 0.6773 | 0.6635 | 0.6773 |

## Classification test BA (forecast h=3.0 s, m=0.5 s)

| cell-mode | rts_light | fb_light | rts_clip | fb_clip |
|---|---|---|---|---|
| PDI1_S2 behavioral | 0.6369 | 0.6354 | 0.6369 | 0.6354 |
| PDI1_S4 behavioral | 0.7694 | 0.7694 | 0.7694 | 0.8028 |
| PDI4_S2 behavioral | 0.6861 | 0.6989 | 0.6861 | 0.6989 |
| PDI4_S3 behavioral | 0.7519 | 0.7650 | 0.7519 | 0.7650 |
| PDI1_S2 laplacian | 0.5595 | 0.5625 | 0.5595 | 0.5625 |
| PDI1_S4 laplacian | 0.7694 | 0.7861 | 0.7694 | 0.7861 |
| PDI4_S2 laplacian | 0.6476 | 0.7161 | 0.6476 | 0.7161 |
| PDI4_S3 laplacian | 0.7387 | 0.7376 | 0.7387 | 0.7376 |

## Classification test BA (forecast h=3.0 s, m=1.0 s)

| cell-mode | rts_light | fb_light | rts_clip | fb_clip |
|---|---|---|---|---|
| PDI1_S2 behavioral | 0.5878 | 0.5893 | 0.5878 | 0.5893 |
| PDI1_S4 behavioral | 0.7694 | 0.7694 | 0.7694 | 0.7694 |
| PDI4_S2 behavioral | 0.6269 | 0.6604 | 0.6269 | 0.6604 |
| PDI4_S3 behavioral | 0.5598 | 0.5932 | 0.5598 | 0.5932 |
| PDI1_S2 laplacian | 0.5640 | 0.5893 | 0.5640 | 0.5893 |
| PDI1_S4 laplacian | 0.7694 | 0.7861 | 0.7694 | 0.7861 |
| PDI4_S2 laplacian | 0.6412 | 0.6484 | 0.6412 | 0.6484 |
| PDI4_S3 laplacian | 0.6844 | 0.6695 | 0.6844 | 0.6695 |
