| module | parameters | layers | heads per layer | width |
|---|---|---|---|---|
| vision encoder | 415 M | 27 | 16 | 1152 |
| language backbone | 2509 M | 18 | 8 | 2048 |
| action expert | 428 M | 18 | 8 | 1024 |
| other (action and time projections) | 2.17 M | | | |

The backbone's count includes its token embedding table (527 M).
Prefix tokens per observation: 3 camera slots x 256 image tokens + 200 prompt tokens = 968. In LIBERO two cameras are real (the third slot is masked) and the prompt is padded, so about 512 image tokens plus the prompt's own tokens are attended to.
Suffix: 10 action tokens, 32 padded action dims, 7 real for LIBERO.
