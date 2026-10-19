# Example test nodes: highest-weighted neighbours (and the lowest-weighted one)

Labels are label indices. True labels are used only for this post-hoc analysis.

### blogcatalog, label_jaccard

Test node 2964 (labels [7, 22], 4 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 2802 | 0.146 | 1/4 | [7] | [7] | 0.50 |
| 2650 | 0.102 | 2/4 | [10] | [] | 0.00 |
| 2227 | 0.079 | 3/4 | [15] | [] | 0.00 |
| 2860 | 0.074 | 4/4 | [5, 7] | [7] | 0.33 |

Test node 5403 (labels [17], 5 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 4801 | 0.111 | 1/5 | [7, 17, 18] | [17] | 0.33 |
| 5926 | 0.093 | 2/5 | [7, 19] | [] | 0.00 |
| 6752 | 0.092 | 3/5 | [3] | [] | 0.00 |
| 4838 | 0.074 | 5/5 | [8] | [] | 0.00 |

Test node 8644 (labels [18], 6 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 8774 | 0.098 | 1/6 | [10, 15, 18] | [18] | 0.33 |
| 9187 | 0.080 | 2/6 | [4] | [] | 0.00 |
| 7148 | 0.073 | 3/6 | [18] | [18] | 1.00 |
| 9 | 0.070 | 6/6 | [4] | [] | 0.00 |

Test node 6528 (labels [27], 7 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 8858 | 0.077 | 1/7 | [5] | [] | 0.00 |
| 2368 | 0.076 | 2/7 | [5, 18, 24, 29, 31] | [] | 0.00 |
| 6065 | 0.074 | 3/7 | [20, 27, 31] | [27] | 0.33 |
| 4983 | 0.068 | 7/7 | [5] | [] | 0.00 |

### blogcatalog, attention

Test node 154 (labels [7], 12 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 175 | 0.077 | 1/12 | [5, 6, 17] | [] | 0.00 |
| 6872 | 0.077 | 2/12 | [22] | [] | 0.00 |
| 6958 | 0.077 | 3/12 | [4] | [] | 0.00 |
| 7010 | 0.077 | 12/12 | [8] | [] | 0.00 |

Test node 8319 (labels [2, 12, 16], 7 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 175 | 0.125 | 1/7 | [5, 6, 17] | [] | 0.00 |
| 4651 | 0.125 | 2/7 | [7] | [] | 0.00 |
| 6958 | 0.125 | 3/7 | [4] | [] | 0.00 |
| 7007 | 0.125 | 7/7 | [23] | [] | 0.00 |

Test node 1946 (labels [7], 12 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 232 | 0.077 | 1/12 | [7] | [7] | 1.00 |
| 644 | 0.077 | 2/12 | [15, 31] | [] | 0.00 |
| 3197 | 0.077 | 3/12 | [1, 2] | [] | 0.00 |
| 8523 | 0.077 | 12/12 | [15] | [] | 0.00 |

Test node 6755 (labels [5], 9 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 448 | 0.100 | 1/9 | [7, 18, 28] | [] | 0.00 |
| 3560 | 0.100 | 2/9 | [21, 22] | [] | 0.00 |
| 175 | 0.100 | 3/9 | [5, 6, 17] | [5] | 0.33 |
| 7934 | 0.100 | 9/9 | [3] | [] | 0.00 |

### pcg, feat_cos

Test node 2400 (labels [14], 9 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 1498 | 0.180 | 1/9 | [7] | [] | 0.00 |
| 620 | 0.129 | 2/9 | [0, 4, 8, 12] | [] | 0.00 |
| 223 | 0.079 | 3/9 | [2, 4, 8] | [] | 0.00 |
| 218 | 0.014 | 9/9 | [8, 13, 14] | [14] | 0.33 |

Test node 3102 (labels [6, 11], 9 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 1823 | 0.133 | 1/9 | [1, 5] | [] | 0.00 |
| 421 | 0.127 | 2/9 | [0, 12] | [] | 0.00 |
| 1763 | 0.108 | 3/9 | [0] | [] | 0.00 |
| 723 | 0.029 | 9/9 | [0, 1, 6, 9, 12, 13] | [6] | 0.14 |

Test node 1859 (labels [0, 4, 8, 12], 11 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 2332 | 0.147 | 1/11 | [8] | [8] | 0.25 |
| 620 | 0.101 | 2/11 | [0, 4, 8, 12] | [0, 4, 8, 12] | 1.00 |
| 2361 | 0.099 | 3/11 | [8] | [8] | 0.25 |
| 1260 | 0.003 | 11/11 | [0, 1, 3, 4, 8, 9, 12, 14] | [0, 4, 8, 12] | 0.50 |

Test node 2160 (labels [4], 4 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 1906 | 0.271 | 1/4 | [1, 6, 7] | [] | 0.00 |
| 1885 | 0.100 | 2/4 | [6] | [] | 0.00 |
| 1618 | 0.072 | 3/4 | [0, 1, 6, 7, 8, 10, 12] | [] | 0.00 |
| 1212 | 0.032 | 4/4 | [1, 6, 7, 8, 12] | [] | 0.00 |

### pcg, label_jaccard

Test node 11 (labels [6], 11 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 1365 | 0.090 | 1/11 | [0, 6, 10, 11, 12] | [6] | 0.20 |
| 1364 | 0.087 | 2/11 | [6, 10, 11, 12] | [6] | 0.25 |
| 1367 | 0.086 | 3/11 | [6, 10, 11, 12] | [6] | 0.25 |
| 1933 | 0.057 | 11/11 | [0] | [] | 0.00 |

Test node 2666 (labels [0], 9 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 2668 | 0.118 | 1/9 | [0] | [0] | 1.00 |
| 2897 | 0.118 | 2/9 | [0, 6] | [0] | 0.50 |
| 2933 | 0.118 | 3/9 | [3] | [] | 0.00 |
| 1910 | 0.051 | 9/9 | [0] | [0] | 1.00 |

Test node 2244 (labels [4, 11, 12], 13 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 3143 | 0.080 | 1/13 | [4] | [4] | 0.33 |
| 2557 | 0.077 | 2/13 | [12] | [12] | 0.33 |
| 736 | 0.069 | 3/13 | [4, 13, 14] | [4] | 0.20 |
| 2486 | 0.051 | 13/13 | [8] | [] | 0.00 |

Test node 1454 (labels [8], 6 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 61 | 0.100 | 1/6 | [0, 2, 3, 7, 9, 12, 13] | [] | 0.00 |
| 447 | 0.082 | 2/6 | [13] | [] | 0.00 |
| 1021 | 0.082 | 3/6 | [9] | [] | 0.00 |
| 2181 | 0.000 | 6/6 | [13] | [] | 0.00 |

### pcg, attention

Test node 2400 (labels [14], 9 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 222 | 0.347 | 1/9 | [0, 4, 11, 12] | [] | 0.00 |
| 620 | 0.330 | 2/9 | [0, 4, 8, 12] | [] | 0.00 |
| 218 | 0.239 | 3/9 | [8, 13, 14] | [14] | 0.33 |
| 224 | 0.000 | 9/9 | [8] | [] | 0.00 |

Test node 650 (labels [8], 8 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 1318 | 0.977 | 1/8 | [2, 9] | [] | 0.00 |
| 2009 | 0.021 | 2/8 | [1, 6, 8] | [8] | 0.33 |
| 2016 | 0.001 | 3/8 | [1] | [] | 0.00 |
| 649 | 0.000 | 8/8 | [0, 1, 4, 6, 8, 12, 13] | [8] | 0.14 |

Test node 2761 (labels [1, 4, 5, 7, 9, 10], 5 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 2112 | 0.737 | 1/5 | [14] | [] | 0.00 |
| 916 | 0.202 | 2/5 | [4] | [4] | 0.17 |
| 2245 | 0.042 | 3/5 | [0, 5, 10] | [5, 10] | 0.29 |
| 182 | 0.002 | 5/5 | [13] | [] | 0.00 |

Test node 2472 (labels [11], 3 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 2042 | 0.925 | 1/3 | [13] | [] | 0.00 |
| 1328 | 0.062 | 2/3 | [9] | [] | 0.00 |
| 1753 | 0.001 | 3/3 | [8] | [] | 0.00 |

### humloc, feat_cos

Test node 2183 (labels [1, 11], 15 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 2979 | 0.082 | 1/15 | [11] | [11] | 0.50 |
| 2979 | 0.082 | 2/15 | [11] | [11] | 0.50 |
| 1041 | 0.069 | 3/15 | [9] | [] | 0.00 |
| 2887 | 0.000 | 15/15 | [10] | [] | 0.00 |

Test node 2534 (labels [7], 15 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 104 | 0.095 | 1/15 | [1] | [] | 0.00 |
| 1731 | 0.093 | 2/15 | [12] | [] | 0.00 |
| 157 | 0.062 | 3/15 | [1] | [] | 0.00 |
| 2863 | 0.008 | 15/15 | [10] | [] | 0.00 |

Test node 2439 (labels [5], 3 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 940 | 0.226 | 1/3 | [6] | [] | 0.00 |
| 90 | 0.160 | 2/3 | [1, 10] | [] | 0.00 |
| 2273 | 0.071 | 3/3 | [4] | [] | 0.00 |

Test node 1957 (labels [1, 4], 11 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 1404 | 0.140 | 1/11 | [10] | [] | 0.00 |
| 121 | 0.096 | 2/11 | [1] | [1] | 0.50 |
| 1496 | 0.085 | 3/11 | [10] | [] | 0.00 |
| 1504 | 0.017 | 11/11 | [10] | [] | 0.00 |

### humloc, label_jaccard

Test node 2704 (labels [10], 15 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 1286 | 0.077 | 1/15 | [10] | [10] | 1.00 |
| 336 | 0.072 | 2/15 | [1, 4] | [] | 0.00 |
| 1566 | 0.069 | 3/15 | [10] | [10] | 1.00 |
| 552 | 0.020 | 15/15 | [2] | [] | 0.00 |

Test node 1866 (labels [0], 11 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 2202 | 0.105 | 1/11 | [1] | [] | 0.00 |
| 2202 | 0.105 | 2/11 | [1] | [] | 0.00 |
| 3105 | 0.098 | 3/11 | [13] | [] | 0.00 |
| 2219 | 0.033 | 11/11 | [2] | [] | 0.00 |

Test node 2008 (labels [1, 10], 15 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 1351 | 0.078 | 1/15 | [10] | [10] | 0.50 |
| 396 | 0.075 | 2/15 | [1, 10] | [1, 10] | 1.00 |
| 1375 | 0.070 | 3/15 | [10] | [10] | 0.50 |
| 988 | 0.032 | 15/15 | [6] | [] | 0.00 |

Test node 1868 (labels [0, 1], 8 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 1888 | 0.134 | 1/8 | [0] | [0] | 0.50 |
| 21 | 0.130 | 2/8 | [0] | [0] | 0.50 |
| 27 | 0.127 | 3/8 | [0] | [0] | 0.50 |
| 2794 | 0.030 | 8/8 | [10] | [] | 0.00 |

### humloc, attention

Test node 2627 (labels [9], 3 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 1393 | 0.327 | 1/3 | [10] | [] | 0.00 |
| 2571 | 0.273 | 2/3 | [9] | [9] | 1.00 |
| 2172 | 0.043 | 3/3 | [1] | [] | 0.00 |

Test node 2797 (labels [10], 10 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 2983 | 0.365 | 1/10 | [11] | [] | 0.00 |
| 185 | 0.198 | 2/10 | [1] | [] | 0.00 |
| 1542 | 0.116 | 3/10 | [10] | [10] | 1.00 |
| 202 | 0.001 | 10/10 | [1, 12] | [] | 0.00 |

Test node 2273 (labels [4], 3 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 2439 | 0.478 | 1/3 | [5] | [] | 0.00 |
| 663 | 0.221 | 2/3 | [4] | [4] | 1.00 |
| 663 | 0.221 | 3/3 | [4] | [4] | 1.00 |

Test node 2395 (labels [5, 12], 9 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 641 | 0.382 | 1/9 | [4] | [] | 0.00 |
| 641 | 0.382 | 2/9 | [4] | [] | 0.00 |
| 380 | 0.077 | 3/9 | [1] | [] | 0.00 |
| 2056 | 0.012 | 9/9 | [1, 10] | [] | 0.00 |

### eukloc, feat_cos

Test node 5946 (labels [6], 4 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 1127 | 0.282 | 1/4 | [6] | [6] | 1.00 |
| 7120 | 0.199 | 2/4 | [17] | [] | 0.00 |
| 5781 | 0.000 | 3/4 | [6, 17] | [6] | 0.50 |
| 6039 | 0.000 | 4/4 | [7] | [] | 0.00 |

Test node 7030 (labels [17], 14 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 1496 | 0.135 | 1/14 | [6] | [] | 0.00 |
| 4032 | 0.106 | 2/14 | [17] | [17] | 1.00 |
| 5485 | 0.076 | 3/14 | [6, 7] | [] | 0.00 |
| 4027 | 0.000 | 14/14 | [17] | [17] | 1.00 |

Test node 7605 (labels [17], 7 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 7141 | 0.149 | 1/7 | [17] | [17] | 1.00 |
| 2610 | 0.138 | 2/7 | [10] | [] | 0.00 |
| 7436 | 0.113 | 3/7 | [17] | [17] | 1.00 |
| 6005 | 0.057 | 7/7 | [6] | [] | 0.00 |

Test node 6952 (labels [16], 13 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 1587 | 0.090 | 1/13 | [6, 17] | [] | 0.00 |
| 5909 | 0.088 | 2/13 | [6, 16] | [16] | 0.50 |
| 4311 | 0.082 | 3/13 | [17] | [] | 0.00 |
| 4307 | 0.023 | 13/13 | [17] | [] | 0.00 |

### eukloc, label_jaccard

Test node 5981 (labels [6, 17], 8 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 492 | 0.151 | 1/8 | [3] | [] | 0.00 |
| 510 | 0.101 | 2/8 | [3, 7] | [] | 0.00 |
| 5918 | 0.095 | 3/8 | [6] | [6] | 0.50 |
| 481 | 0.039 | 8/8 | [3] | [] | 0.00 |

Test node 7438 (labels [17], 7 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 805 | 0.130 | 1/7 | [6, 17] | [17] | 0.50 |
| 7302 | 0.124 | 2/7 | [17] | [17] | 1.00 |
| 263 | 0.124 | 3/7 | [1, 6] | [] | 0.00 |
| 5377 | 0.071 | 7/7 | [6] | [] | 0.00 |

Test node 6768 (labels [14], 13 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 6955 | 0.093 | 1/13 | [16] | [] | 0.00 |
| 4307 | 0.092 | 2/13 | [17] | [] | 0.00 |
| 5990 | 0.091 | 3/13 | [6] | [] | 0.00 |
| 4964 | 0.021 | 13/13 | [3, 6] | [] | 0.00 |

Test node 5309 (labels [6], 4 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 4370 | 0.187 | 1/4 | [17] | [] | 0.00 |
| 5353 | 0.183 | 2/4 | [6] | [6] | 1.00 |
| 5353 | 0.183 | 3/4 | [6] | [6] | 1.00 |
| 1853 | 0.167 | 4/4 | [6] | [6] | 1.00 |

### eukloc, attention

Test node 5766 (labels [6, 17], 11 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 5323 | 0.430 | 1/11 | [6] | [6] | 0.50 |
| 5876 | 0.215 | 2/11 | [6] | [6] | 0.50 |
| 7161 | 0.099 | 3/11 | [17] | [17] | 0.50 |
| 4947 | 0.008 | 11/11 | [2] | [] | 0.00 |

Test node 5934 (labels [6], 4 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 3876 | 0.648 | 1/4 | [17] | [] | 0.00 |
| 4056 | 0.163 | 2/4 | [17] | [] | 0.00 |
| 853 | 0.140 | 3/4 | [6] | [6] | 1.00 |
| 1873 | 0.024 | 4/4 | [6] | [6] | 1.00 |

Test node 6222 (labels [8], 9 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 2278 | 0.150 | 1/9 | [8] | [8] | 1.00 |
| 2278 | 0.150 | 2/9 | [8] | [8] | 1.00 |
| 2160 | 0.120 | 3/9 | [8] | [8] | 1.00 |
| 2179 | 0.015 | 9/9 | [8] | [8] | 1.00 |

Test node 6059 (labels [7], 7 neighbours):

| neighbour | weight | rank | its labels | shared | Jaccard |
|---|---|---|---|---|---|
| 2054 | 0.443 | 1/7 | [6] | [] | 0.00 |
| 2071 | 0.244 | 2/7 | [7] | [7] | 1.00 |
| 6031 | 0.139 | 3/7 | [7] | [7] | 1.00 |
| 2075 | 0.001 | 7/7 | [7] | [7] | 1.00 |

