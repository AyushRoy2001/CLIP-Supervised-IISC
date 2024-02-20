# CLIP-Supervised-IISC

|             Experimental Setup                                                 | SROCC |
| ------------------------------------------------------------------------------ | ----- |
| Standard transformation of images                                                      |
| default                                                                        | 69.69 |
| scaling-0.1                                                                    | 70.38 |
| scaling-0.1, batch_size-8                                                      | 66    |
| scaling-0.1, random_samples-12                                                 | 60.48 |
| scaling-0.1, alpha-1.0                                                         | 68.68 |
| scaling-0.1, eye*0.3                                                           | 74.54 |  
| scaling-0.1, eye*0.1                                                           | 77.50 |
| scaling-0.1, eye*0.05                                                          | 79.16 |
| scaling-0.1, eye*0.01                                                          | 82.29 |
| scaling-0.1, eye*0.005                                                         | 80.38 |
| scaling-0.1, eye*0.0075                                                        | 77.18 |
| scaling-0.1, eye*0.01, batch_size-8, random_samples-16                         | 79.34 |
| scaling-0.1, eye*0.01, alpha-0.6                                               | 76.39 |
| scaling-0.1, eye*0.01, alpha-0.4                                               | 78.89 |
| CLIP IQA's transform                                                                   |
| scaling-0.1, eye*0.01                                                          | 73.62 |
| scaling-0.1, batch_size-8, eye*0.01                                            | 79.72 |
| scaling-0.1, batch_size-12, eye*0.01                                           | 77.33 |
| scaling-0.1, batch_size-8, eye*0.01, random_samples-12                         | 82.35 |
