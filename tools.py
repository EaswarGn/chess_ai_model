FILES = ["a", "b", "c", "d", "e", "f", "g", "h"]
RANKS = ["1", "2", "3", "4", "5", "6", "7", "8"]
PIECES = {
    ".": 0,
    ",": 1,
    "P": 2,
    "p": 3,
    "R": 4,
    "r": 5,
    "N": 6,
    "n": 7,
    "B": 8,
    "b": 9,
    "Q": 10,
    "q": 11,
    "K": 12,
    "k": 13,
}
TURN = {"b": 0, "w": 1}
SQUARES = {
    "a8": 0,
    "b8": 1,
    "c8": 2,
    "d8": 3,
    "e8": 4,
    "f8": 5,
    "g8": 6,
    "h8": 7,
    "a7": 8,
    "b7": 9,
    "c7": 10,
    "d7": 11,
    "e7": 12,
    "f7": 13,
    "g7": 14,
    "h7": 15,
    "a6": 16,
    "b6": 17,
    "c6": 18,
    "d6": 19,
    "e6": 20,
    "f6": 21,
    "g6": 22,
    "h6": 23,
    "a5": 24,
    "b5": 25,
    "c5": 26,
    "d5": 27,
    "e5": 28,
    "f5": 29,
    "g5": 30,
    "h5": 31,
    "a4": 32,
    "b4": 33,
    "c4": 34,
    "d4": 35,
    "e4": 36,
    "f4": 37,
    "g4": 38,
    "h4": 39,
    "a3": 40,
    "b3": 41,
    "c3": 42,
    "d3": 43,
    "e3": 44,
    "f3": 45,
    "g3": 46,
    "h3": 47,
    "a2": 48,
    "b2": 49,
    "c2": 50,
    "d2": 51,
    "e2": 52,
    "f2": 53,
    "g2": 54,
    "h2": 55,
    "a1": 56,
    "b1": 57,
    "c1": 58,
    "d1": 59,
    "e1": 60,
    "f1": 61,
    "g1": 62,
    "h1": 63,
}
UCI_MOVES = {
    "a1h8": 0,
    "a1a8": 1,
    "a1g7": 2,
    "a1a7": 3,
    "a1f6": 4,
    "a1a6": 5,
    "a1e5": 6,
    "a1a5": 7,
    "a1d4": 8,
    "a1a4": 9,
    "a1c3": 10,
    "a1a3": 11,
    "a1b2": 12,
    "a1a2": 13,
    "a1h1": 14,
    "a1g1": 15,
    "a1f1": 16,
    "a1e1": 17,
    "a1d1": 18,
    "a1c1": 19,
    "a1b1": 20,
    "a2g8": 21,
    "a2a8": 22,
    "a2f7": 23,
    "a2a7": 24,
    "a2e6": 25,
    "a2a6": 26,
    "a2d5": 27,
    "a2a5": 28,
    "a2c4": 29,
    "a2a4": 30,
    "a2b3": 31,
    "a2a3": 32,
    "a2h2": 33,
    "a2g2": 34,
    "a2f2": 35,
    "a2e2": 36,
    "a2d2": 37,
    "a2c2": 38,
    "a2b2": 39,
    "a2b1": 40,
    "a2a1": 41,
    "a3f8": 42,
    "a3a8": 43,
    "a3e7": 44,
    "a3a7": 45,
    "a3d6": 46,
    "a3a6": 47,
    "a3c5": 48,
    "a3a5": 49,
    "a3b4": 50,
    "a3a4": 51,
    "a3h3": 52,
    "a3g3": 53,
    "a3f3": 54,
    "a3e3": 55,
    "a3d3": 56,
    "a3c3": 57,
    "a3b3": 58,
    "a3b2": 59,
    "a3a2": 60,
    "a3c1": 61,
    "a3a1": 62,
    "a4e8": 63,
    "a4a8": 64,
    "a4d7": 65,
    "a4a7": 66,
    "a4c6": 67,
    "a4a6": 68,
    "a4b5": 69,
    "a4a5": 70,
    "a4h4": 71,
    "a4g4": 72,
    "a4f4": 73,
    "a4e4": 74,
    "a4d4": 75,
    "a4c4": 76,
    "a4b4": 77,
    "a4b3": 78,
    "a4a3": 79,
    "a4c2": 80,
    "a4a2": 81,
    "a4d1": 82,
    "a4a1": 83,
    "a5d8": 84,
    "a5a8": 85,
    "a5c7": 86,
    "a5a7": 87,
    "a5b6": 88,
    "a5a6": 89,
    "a5h5": 90,
    "a5g5": 91,
    "a5f5": 92,
    "a5e5": 93,
    "a5d5": 94,
    "a5c5": 95,
    "a5b5": 96,
    "a5b4": 97,
    "a5a4": 98,
    "a5c3": 99,
    "a5a3": 100,
    "a5d2": 101,
    "a5a2": 102,
    "a5e1": 103,
    "a5a1": 104,
    "a6c8": 105,
    "a6a8": 106,
    "a6b7": 107,
    "a6a7": 108,
    "a6h6": 109,
    "a6g6": 110,
    "a6f6": 111,
    "a6e6": 112,
    "a6d6": 113,
    "a6c6": 114,
    "a6b6": 115,
    "a6b5": 116,
    "a6a5": 117,
    "a6c4": 118,
    "a6a4": 119,
    "a6d3": 120,
    "a6a3": 121,
    "a6e2": 122,
    "a6a2": 123,
    "a6f1": 124,
    "a6a1": 125,
    "a7b8": 126,
    "a7a8": 127,
    "a7h7": 128,
    "a7g7": 129,
    "a7f7": 130,
    "a7e7": 131,
    "a7d7": 132,
    "a7c7": 133,
    "a7b7": 134,
    "a7b6": 135,
    "a7a6": 136,
    "a7c5": 137,
    "a7a5": 138,
    "a7d4": 139,
    "a7a4": 140,
    "a7e3": 141,
    "a7a3": 142,
    "a7f2": 143,
    "a7a2": 144,
    "a7g1": 145,
    "a7a1": 146,
    "a8h8": 147,
    "a8g8": 148,
    "a8f8": 149,
    "a8e8": 150,
    "a8d8": 151,
    "a8c8": 152,
    "a8b8": 153,
    "a8b7": 154,
    "a8a7": 155,
    "a8c6": 156,
    "a8a6": 157,
    "a8d5": 158,
    "a8a5": 159,
    "a8e4": 160,
    "a8a4": 161,
    "a8f3": 162,
    "a8a3": 163,
    "a8g2": 164,
    "a8a2": 165,
    "a8h1": 166,
    "a8a1": 167,
    "b1b8": 168,
    "b1h7": 169,
    "b1b7": 170,
    "b1g6": 171,
    "b1b6": 172,
    "b1f5": 173,
    "b1b5": 174,
    "b1e4": 175,
    "b1b4": 176,
    "b1d3": 177,
    "b1b3": 178,
    "b1c2": 179,
    "b1b2": 180,
    "b1a2": 181,
    "b1h1": 182,
    "b1g1": 183,
    "b1f1": 184,
    "b1e1": 185,
    "b1d1": 186,
    "b1c1": 187,
    "b1a1": 188,
    "b2h8": 189,
    "b2b8": 190,
    "b2g7": 191,
    "b2b7": 192,
    "b2f6": 193,
    "b2b6": 194,
    "b2e5": 195,
    "b2b5": 196,
    "b2d4": 197,
    "b2b4": 198,
    "b2c3": 199,
    "b2b3": 200,
    "b2a3": 201,
    "b2h2": 202,
    "b2g2": 203,
    "b2f2": 204,
    "b2e2": 205,
    "b2d2": 206,
    "b2c2": 207,
    "b2a2": 208,
    "b2c1": 209,
    "b2b1": 210,
    "b2a1": 211,
    "b3g8": 212,
    "b3b8": 213,
    "b3f7": 214,
    "b3b7": 215,
    "b3e6": 216,
    "b3b6": 217,
    "b3d5": 218,
    "b3b5": 219,
    "b3c4": 220,
    "b3b4": 221,
    "b3a4": 222,
    "b3h3": 223,
    "b3g3": 224,
    "b3f3": 225,
    "b3e3": 226,
    "b3d3": 227,
    "b3c3": 228,
    "b3a3": 229,
    "b3c2": 230,
    "b3b2": 231,
    "b3a2": 232,
    "b3d1": 233,
    "b3b1": 234,
    "b4f8": 235,
    "b4b8": 236,
    "b4e7": 237,
    "b4b7": 238,
    "b4d6": 239,
    "b4b6": 240,
    "b4c5": 241,
    "b4b5": 242,
    "b4a5": 243,
    "b4h4": 244,
    "b4g4": 245,
    "b4f4": 246,
    "b4e4": 247,
    "b4d4": 248,
    "b4c4": 249,
    "b4a4": 250,
    "b4c3": 251,
    "b4b3": 252,
    "b4a3": 253,
    "b4d2": 254,
    "b4b2": 255,
    "b4e1": 256,
    "b4b1": 257,
    "b5e8": 258,
    "b5b8": 259,
    "b5d7": 260,
    "b5b7": 261,
    "b5c6": 262,
    "b5b6": 263,
    "b5a6": 264,
    "b5h5": 265,
    "b5g5": 266,
    "b5f5": 267,
    "b5e5": 268,
    "b5d5": 269,
    "b5c5": 270,
    "b5a5": 271,
    "b5c4": 272,
    "b5b4": 273,
    "b5a4": 274,
    "b5d3": 275,
    "b5b3": 276,
    "b5e2": 277,
    "b5b2": 278,
    "b5f1": 279,
    "b5b1": 280,
    "b6d8": 281,
    "b6b8": 282,
    "b6c7": 283,
    "b6b7": 284,
    "b6a7": 285,
    "b6h6": 286,
    "b6g6": 287,
    "b6f6": 288,
    "b6e6": 289,
    "b6d6": 290,
    "b6c6": 291,
    "b6a6": 292,
    "b6c5": 293,
    "b6b5": 294,
    "b6a5": 295,
    "b6d4": 296,
    "b6b4": 297,
    "b6e3": 298,
    "b6b3": 299,
    "b6f2": 300,
    "b6b2": 301,
    "b6g1": 302,
    "b6b1": 303,
    "b7c8": 304,
    "b7b8": 305,
    "b7a8": 306,
    "b7h7": 307,
    "b7g7": 308,
    "b7f7": 309,
    "b7e7": 310,
    "b7d7": 311,
    "b7c7": 312,
    "b7a7": 313,
    "b7c6": 314,
    "b7b6": 315,
    "b7a6": 316,
    "b7d5": 317,
    "b7b5": 318,
    "b7e4": 319,
    "b7b4": 320,
    "b7f3": 321,
    "b7b3": 322,
    "b7g2": 323,
    "b7b2": 324,
    "b7h1": 325,
    "b7b1": 326,
    "b8h8": 327,
    "b8g8": 328,
    "b8f8": 329,
    "b8e8": 330,
    "b8d8": 331,
    "b8c8": 332,
    "b8a8": 333,
    "b8c7": 334,
    "b8b7": 335,
    "b8a7": 336,
    "b8d6": 337,
    "b8b6": 338,
    "b8e5": 339,
    "b8b5": 340,
    "b8f4": 341,
    "b8b4": 342,
    "b8g3": 343,
    "b8b3": 344,
    "b8h2": 345,
    "b8b2": 346,
    "b8b1": 347,
    "c1c8": 348,
    "c1c7": 349,
    "c1h6": 350,
    "c1c6": 351,
    "c1g5": 352,
    "c1c5": 353,
    "c1f4": 354,
    "c1c4": 355,
    "c1e3": 356,
    "c1c3": 357,
    "c1a3": 358,
    "c1d2": 359,
    "c1c2": 360,
    "c1b2": 361,
    "c1h1": 362,
    "c1g1": 363,
    "c1f1": 364,
    "c1e1": 365,
    "c1d1": 366,
    "c1b1": 367,
    "c1a1": 368,
    "c2c8": 369,
    "c2h7": 370,
    "c2c7": 371,
    "c2g6": 372,
    "c2c6": 373,
    "c2f5": 374,
    "c2c5": 375,
    "c2e4": 376,
    "c2c4": 377,
    "c2a4": 378,
    "c2d3": 379,
    "c2c3": 380,
    "c2b3": 381,
    "c2h2": 382,
    "c2g2": 383,
    "c2f2": 384,
    "c2e2": 385,
    "c2d2": 386,
    "c2b2": 387,
    "c2a2": 388,
    "c2d1": 389,
    "c2c1": 390,
    "c2b1": 391,
    "c3h8": 392,
    "c3c8": 393,
    "c3g7": 394,
    "c3c7": 395,
    "c3f6": 396,
    "c3c6": 397,
    "c3e5": 398,
    "c3c5": 399,
    "c3a5": 400,
    "c3d4": 401,
    "c3c4": 402,
    "c3b4": 403,
    "c3h3": 404,
    "c3g3": 405,
    "c3f3": 406,
    "c3e3": 407,
    "c3d3": 408,
    "c3b3": 409,
    "c3a3": 410,
    "c3d2": 411,
    "c3c2": 412,
    "c3b2": 413,
    "c3e1": 414,
    "c3c1": 415,
    "c3a1": 416,
    "c4g8": 417,
    "c4c8": 418,
    "c4f7": 419,
    "c4c7": 420,
    "c4e6": 421,
    "c4c6": 422,
    "c4a6": 423,
    "c4d5": 424,
    "c4c5": 425,
    "c4b5": 426,
    "c4h4": 427,
    "c4g4": 428,
    "c4f4": 429,
    "c4e4": 430,
    "c4d4": 431,
    "c4b4": 432,
    "c4a4": 433,
    "c4d3": 434,
    "c4c3": 435,
    "c4b3": 436,
    "c4e2": 437,
    "c4c2": 438,
    "c4a2": 439,
    "c4f1": 440,
    "c4c1": 441,
    "c5f8": 442,
    "c5c8": 443,
    "c5e7": 444,
    "c5c7": 445,
    "c5a7": 446,
    "c5d6": 447,
    "c5c6": 448,
    "c5b6": 449,
    "c5h5": 450,
    "c5g5": 451,
    "c5f5": 452,
    "c5e5": 453,
    "c5d5": 454,
    "c5b5": 455,
    "c5a5": 456,
    "c5d4": 457,
    "c5c4": 458,
    "c5b4": 459,
    "c5e3": 460,
    "c5c3": 461,
    "c5a3": 462,
    "c5f2": 463,
    "c5c2": 464,
    "c5g1": 465,
    "c5c1": 466,
    "c6e8": 467,
    "c6c8": 468,
    "c6a8": 469,
    "c6d7": 470,
    "c6c7": 471,
    "c6b7": 472,
    "c6h6": 473,
    "c6g6": 474,
    "c6f6": 475,
    "c6e6": 476,
    "c6d6": 477,
    "c6b6": 478,
    "c6a6": 479,
    "c6d5": 480,
    "c6c5": 481,
    "c6b5": 482,
    "c6e4": 483,
    "c6c4": 484,
    "c6a4": 485,
    "c6f3": 486,
    "c6c3": 487,
    "c6g2": 488,
    "c6c2": 489,
    "c6h1": 490,
    "c6c1": 491,
    "c7d8": 492,
    "c7c8": 493,
    "c7b8": 494,
    "c7h7": 495,
    "c7g7": 496,
    "c7f7": 497,
    "c7e7": 498,
    "c7d7": 499,
    "c7b7": 500,
    "c7a7": 501,
    "c7d6": 502,
    "c7c6": 503,
    "c7b6": 504,
    "c7e5": 505,
    "c7c5": 506,
    "c7a5": 507,
    "c7f4": 508,
    "c7c4": 509,
    "c7g3": 510,
    "c7c3": 511,
    "c7h2": 512,
    "c7c2": 513,
    "c7c1": 514,
    "c8h8": 515,
    "c8g8": 516,
    "c8f8": 517,
    "c8e8": 518,
    "c8d8": 519,
    "c8b8": 520,
    "c8a8": 521,
    "c8d7": 522,
    "c8c7": 523,
    "c8b7": 524,
    "c8e6": 525,
    "c8c6": 526,
    "c8a6": 527,
    "c8f5": 528,
    "c8c5": 529,
    "c8g4": 530,
    "c8c4": 531,
    "c8h3": 532,
    "c8c3": 533,
    "c8c2": 534,
    "c8c1": 535,
    "d1d8": 536,
    "d1d7": 537,
    "d1d6": 538,
    "d1h5": 539,
    "d1d5": 540,
    "d1g4": 541,
    "d1d4": 542,
    "d1a4": 543,
    "d1f3": 544,
    "d1d3": 545,
    "d1b3": 546,
    "d1e2": 547,
    "d1d2": 548,
    "d1c2": 549,
    "d1h1": 550,
    "d1g1": 551,
    "d1f1": 552,
    "d1e1": 553,
    "d1c1": 554,
    "d1b1": 555,
    "d1a1": 556,
    "d2d8": 557,
    "d2d7": 558,
    "d2h6": 559,
    "d2d6": 560,
    "d2g5": 561,
    "d2d5": 562,
    "d2a5": 563,
    "d2f4": 564,
    "d2d4": 565,
    "d2b4": 566,
    "d2e3": 567,
    "d2d3": 568,
    "d2c3": 569,
    "d2h2": 570,
    "d2g2": 571,
    "d2f2": 572,
    "d2e2": 573,
    "d2c2": 574,
    "d2b2": 575,
    "d2a2": 576,
    "d2e1": 577,
    "d2d1": 578,
    "d2c1": 579,
    "d3d8": 580,
    "d3h7": 581,
    "d3d7": 582,
    "d3g6": 583,
    "d3d6": 584,
    "d3a6": 585,
    "d3f5": 586,
    "d3d5": 587,
    "d3b5": 588,
    "d3e4": 589,
    "d3d4": 590,
    "d3c4": 591,
    "d3h3": 592,
    "d3g3": 593,
    "d3f3": 594,
    "d3e3": 595,
    "d3c3": 596,
    "d3b3": 597,
    "d3a3": 598,
    "d3e2": 599,
    "d3d2": 600,
    "d3c2": 601,
    "d3f1": 602,
    "d3d1": 603,
    "d3b1": 604,
    "d4h8": 605,
    "d4d8": 606,
    "d4g7": 607,
    "d4d7": 608,
    "d4a7": 609,
    "d4f6": 610,
    "d4d6": 611,
    "d4b6": 612,
    "d4e5": 613,
    "d4d5": 614,
    "d4c5": 615,
    "d4h4": 616,
    "d4g4": 617,
    "d4f4": 618,
    "d4e4": 619,
    "d4c4": 620,
    "d4b4": 621,
    "d4a4": 622,
    "d4e3": 623,
    "d4d3": 624,
    "d4c3": 625,
    "d4f2": 626,
    "d4d2": 627,
    "d4b2": 628,
    "d4g1": 629,
    "d4d1": 630,
    "d4a1": 631,
    "d5g8": 632,
    "d5d8": 633,
    "d5a8": 634,
    "d5f7": 635,
    "d5d7": 636,
    "d5b7": 637,
    "d5e6": 638,
    "d5d6": 639,
    "d5c6": 640,
    "d5h5": 641,
    "d5g5": 642,
    "d5f5": 643,
    "d5e5": 644,
    "d5c5": 645,
    "d5b5": 646,
    "d5a5": 647,
    "d5e4": 648,
    "d5d4": 649,
    "d5c4": 650,
    "d5f3": 651,
    "d5d3": 652,
    "d5b3": 653,
    "d5g2": 654,
    "d5d2": 655,
    "d5a2": 656,
    "d5h1": 657,
    "d5d1": 658,
    "d6f8": 659,
    "d6d8": 660,
    "d6b8": 661,
    "d6e7": 662,
    "d6d7": 663,
    "d6c7": 664,
    "d6h6": 665,
    "d6g6": 666,
    "d6f6": 667,
    "d6e6": 668,
    "d6c6": 669,
    "d6b6": 670,
    "d6a6": 671,
    "d6e5": 672,
    "d6d5": 673,
    "d6c5": 674,
    "d6f4": 675,
    "d6d4": 676,
    "d6b4": 677,
    "d6g3": 678,
    "d6d3": 679,
    "d6a3": 680,
    "d6h2": 681,
    "d6d2": 682,
    "d6d1": 683,
    "d7e8": 684,
    "d7d8": 685,
    "d7c8": 686,
    "d7h7": 687,
    "d7g7": 688,
    "d7f7": 689,
    "d7e7": 690,
    "d7c7": 691,
    "d7b7": 692,
    "d7a7": 693,
    "d7e6": 694,
    "d7d6": 695,
    "d7c6": 696,
    "d7f5": 697,
    "d7d5": 698,
    "d7b5": 699,
    "d7g4": 700,
    "d7d4": 701,
    "d7a4": 702,
    "d7h3": 703,
    "d7d3": 704,
    "d7d2": 705,
    "d7d1": 706,
    "d8h8": 707,
    "d8g8": 708,
    "d8f8": 709,
    "d8e8": 710,
    "d8c8": 711,
    "d8b8": 712,
    "d8a8": 713,
    "d8e7": 714,
    "d8d7": 715,
    "d8c7": 716,
    "d8f6": 717,
    "d8d6": 718,
    "d8b6": 719,
    "d8g5": 720,
    "d8d5": 721,
    "d8a5": 722,
    "d8h4": 723,
    "d8d4": 724,
    "d8d3": 725,
    "d8d2": 726,
    "d8d1": 727,
    "e1e8": 728,
    "e1e7": 729,
    "e1e6": 730,
    "e1e5": 731,
    "e1a5": 732,
    "e1h4": 733,
    "e1e4": 734,
    "e1b4": 735,
    "e1g3": 736,
    "e1e3": 737,
    "e1c3": 738,
    "e1f2": 739,
    "e1e2": 740,
    "e1d2": 741,
    "e1h1": 742,
    "e1g1": 743,
    "e1f1": 744,
    "e1d1": 745,
    "e1c1": 746,
    "e1b1": 747,
    "e1a1": 748,
    "e2e8": 749,
    "e2e7": 750,
    "e2e6": 751,
    "e2a6": 752,
    "e2h5": 753,
    "e2e5": 754,
    "e2b5": 755,
    "e2g4": 756,
    "e2e4": 757,
    "e2c4": 758,
    "e2f3": 759,
    "e2e3": 760,
    "e2d3": 761,
    "e2h2": 762,
    "e2g2": 763,
    "e2f2": 764,
    "e2d2": 765,
    "e2c2": 766,
    "e2b2": 767,
    "e2a2": 768,
    "e2f1": 769,
    "e2e1": 770,
    "e2d1": 771,
    "e3e8": 772,
    "e3e7": 773,
    "e3a7": 774,
    "e3h6": 775,
    "e3e6": 776,
    "e3b6": 777,
    "e3g5": 778,
    "e3e5": 779,
    "e3c5": 780,
    "e3f4": 781,
    "e3e4": 782,
    "e3d4": 783,
    "e3h3": 784,
    "e3g3": 785,
    "e3f3": 786,
    "e3d3": 787,
    "e3c3": 788,
    "e3b3": 789,
    "e3a3": 790,
    "e3f2": 791,
    "e3e2": 792,
    "e3d2": 793,
    "e3g1": 794,
    "e3e1": 795,
    "e3c1": 796,
    "e4e8": 797,
    "e4a8": 798,
    "e4h7": 799,
    "e4e7": 800,
    "e4b7": 801,
    "e4g6": 802,
    "e4e6": 803,
    "e4c6": 804,
    "e4f5": 805,
    "e4e5": 806,
    "e4d5": 807,
    "e4h4": 808,
    "e4g4": 809,
    "e4f4": 810,
    "e4d4": 811,
    "e4c4": 812,
    "e4b4": 813,
    "e4a4": 814,
    "e4f3": 815,
    "e4e3": 816,
    "e4d3": 817,
    "e4g2": 818,
    "e4e2": 819,
    "e4c2": 820,
    "e4h1": 821,
    "e4e1": 822,
    "e4b1": 823,
    "e5h8": 824,
    "e5e8": 825,
    "e5b8": 826,
    "e5g7": 827,
    "e5e7": 828,
    "e5c7": 829,
    "e5f6": 830,
    "e5e6": 831,
    "e5d6": 832,
    "e5h5": 833,
    "e5g5": 834,
    "e5f5": 835,
    "e5d5": 836,
    "e5c5": 837,
    "e5b5": 838,
    "e5a5": 839,
    "e5f4": 840,
    "e5e4": 841,
    "e5d4": 842,
    "e5g3": 843,
    "e5e3": 844,
    "e5c3": 845,
    "e5h2": 846,
    "e5e2": 847,
    "e5b2": 848,
    "e5e1": 849,
    "e5a1": 850,
    "e6g8": 851,
    "e6e8": 852,
    "e6c8": 853,
    "e6f7": 854,
    "e6e7": 855,
    "e6d7": 856,
    "e6h6": 857,
    "e6g6": 858,
    "e6f6": 859,
    "e6d6": 860,
    "e6c6": 861,
    "e6b6": 862,
    "e6a6": 863,
    "e6f5": 864,
    "e6e5": 865,
    "e6d5": 866,
    "e6g4": 867,
    "e6e4": 868,
    "e6c4": 869,
    "e6h3": 870,
    "e6e3": 871,
    "e6b3": 872,
    "e6e2": 873,
    "e6a2": 874,
    "e6e1": 875,
    "e7f8": 876,
    "e7e8": 877,
    "e7d8": 878,
    "e7h7": 879,
    "e7g7": 880,
    "e7f7": 881,
    "e7d7": 882,
    "e7c7": 883,
    "e7b7": 884,
    "e7a7": 885,
    "e7f6": 886,
    "e7e6": 887,
    "e7d6": 888,
    "e7g5": 889,
    "e7e5": 890,
    "e7c5": 891,
    "e7h4": 892,
    "e7e4": 893,
    "e7b4": 894,
    "e7e3": 895,
    "e7a3": 896,
    "e7e2": 897,
    "e7e1": 898,
    "e8h8": 899,
    "e8g8": 900,
    "e8f8": 901,
    "e8d8": 902,
    "e8c8": 903,
    "e8b8": 904,
    "e8a8": 905,
    "e8f7": 906,
    "e8e7": 907,
    "e8d7": 908,
    "e8g6": 909,
    "e8e6": 910,
    "e8c6": 911,
    "e8h5": 912,
    "e8e5": 913,
    "e8b5": 914,
    "e8e4": 915,
    "e8a4": 916,
    "e8e3": 917,
    "e8e2": 918,
    "e8e1": 919,
    "f1f8": 920,
    "f1f7": 921,
    "f1f6": 922,
    "f1a6": 923,
    "f1f5": 924,
    "f1b5": 925,
    "f1f4": 926,
    "f1c4": 927,
    "f1h3": 928,
    "f1f3": 929,
    "f1d3": 930,
    "f1g2": 931,
    "f1f2": 932,
    "f1e2": 933,
    "f1h1": 934,
    "f1g1": 935,
    "f1e1": 936,
    "f1d1": 937,
    "f1c1": 938,
    "f1b1": 939,
    "f1a1": 940,
    "f2f8": 941,
    "f2f7": 942,
    "f2a7": 943,
    "f2f6": 944,
    "f2b6": 945,
    "f2f5": 946,
    "f2c5": 947,
    "f2h4": 948,
    "f2f4": 949,
    "f2d4": 950,
    "f2g3": 951,
    "f2f3": 952,
    "f2e3": 953,
    "f2h2": 954,
    "f2g2": 955,
    "f2e2": 956,
    "f2d2": 957,
    "f2c2": 958,
    "f2b2": 959,
    "f2a2": 960,
    "f2g1": 961,
    "f2f1": 962,
    "f2e1": 963,
    "f3f8": 964,
    "f3a8": 965,
    "f3f7": 966,
    "f3b7": 967,
    "f3f6": 968,
    "f3c6": 969,
    "f3h5": 970,
    "f3f5": 971,
    "f3d5": 972,
    "f3g4": 973,
    "f3f4": 974,
    "f3e4": 975,
    "f3h3": 976,
    "f3g3": 977,
    "f3e3": 978,
    "f3d3": 979,
    "f3c3": 980,
    "f3b3": 981,
    "f3a3": 982,
    "f3g2": 983,
    "f3f2": 984,
    "f3e2": 985,
    "f3h1": 986,
    "f3f1": 987,
    "f3d1": 988,
    "f4f8": 989,
    "f4b8": 990,
    "f4f7": 991,
    "f4c7": 992,
    "f4h6": 993,
    "f4f6": 994,
    "f4d6": 995,
    "f4g5": 996,
    "f4f5": 997,
    "f4e5": 998,
    "f4h4": 999,
    "f4g4": 1000,
    "f4e4": 1001,
    "f4d4": 1002,
    "f4c4": 1003,
    "f4b4": 1004,
    "f4a4": 1005,
    "f4g3": 1006,
    "f4f3": 1007,
    "f4e3": 1008,
    "f4h2": 1009,
    "f4f2": 1010,
    "f4d2": 1011,
    "f4f1": 1012,
    "f4c1": 1013,
    "f5f8": 1014,
    "f5c8": 1015,
    "f5h7": 1016,
    "f5f7": 1017,
    "f5d7": 1018,
    "f5g6": 1019,
    "f5f6": 1020,
    "f5e6": 1021,
    "f5h5": 1022,
    "f5g5": 1023,
    "f5e5": 1024,
    "f5d5": 1025,
    "f5c5": 1026,
    "f5b5": 1027,
    "f5a5": 1028,
    "f5g4": 1029,
    "f5f4": 1030,
    "f5e4": 1031,
    "f5h3": 1032,
    "f5f3": 1033,
    "f5d3": 1034,
    "f5f2": 1035,
    "f5c2": 1036,
    "f5f1": 1037,
    "f5b1": 1038,
    "f6h8": 1039,
    "f6f8": 1040,
    "f6d8": 1041,
    "f6g7": 1042,
    "f6f7": 1043,
    "f6e7": 1044,
    "f6h6": 1045,
    "f6g6": 1046,
    "f6e6": 1047,
    "f6d6": 1048,
    "f6c6": 1049,
    "f6b6": 1050,
    "f6a6": 1051,
    "f6g5": 1052,
    "f6f5": 1053,
    "f6e5": 1054,
    "f6h4": 1055,
    "f6f4": 1056,
    "f6d4": 1057,
    "f6f3": 1058,
    "f6c3": 1059,
    "f6f2": 1060,
    "f6b2": 1061,
    "f6f1": 1062,
    "f6a1": 1063,
    "f7g8": 1064,
    "f7f8": 1065,
    "f7e8": 1066,
    "f7h7": 1067,
    "f7g7": 1068,
    "f7e7": 1069,
    "f7d7": 1070,
    "f7c7": 1071,
    "f7b7": 1072,
    "f7a7": 1073,
    "f7g6": 1074,
    "f7f6": 1075,
    "f7e6": 1076,
    "f7h5": 1077,
    "f7f5": 1078,
    "f7d5": 1079,
    "f7f4": 1080,
    "f7c4": 1081,
    "f7f3": 1082,
    "f7b3": 1083,
    "f7f2": 1084,
    "f7a2": 1085,
    "f7f1": 1086,
    "f8h8": 1087,
    "f8g8": 1088,
    "f8e8": 1089,
    "f8d8": 1090,
    "f8c8": 1091,
    "f8b8": 1092,
    "f8a8": 1093,
    "f8g7": 1094,
    "f8f7": 1095,
    "f8e7": 1096,
    "f8h6": 1097,
    "f8f6": 1098,
    "f8d6": 1099,
    "f8f5": 1100,
    "f8c5": 1101,
    "f8f4": 1102,
    "f8b4": 1103,
    "f8f3": 1104,
    "f8a3": 1105,
    "f8f2": 1106,
    "f8f1": 1107,
    "g1g8": 1108,
    "g1g7": 1109,
    "g1a7": 1110,
    "g1g6": 1111,
    "g1b6": 1112,
    "g1g5": 1113,
    "g1c5": 1114,
    "g1g4": 1115,
    "g1d4": 1116,
    "g1g3": 1117,
    "g1e3": 1118,
    "g1h2": 1119,
    "g1g2": 1120,
    "g1f2": 1121,
    "g1h1": 1122,
    "g1f1": 1123,
    "g1e1": 1124,
    "g1d1": 1125,
    "g1c1": 1126,
    "g1b1": 1127,
    "g1a1": 1128,
    "g2g8": 1129,
    "g2a8": 1130,
    "g2g7": 1131,
    "g2b7": 1132,
    "g2g6": 1133,
    "g2c6": 1134,
    "g2g5": 1135,
    "g2d5": 1136,
    "g2g4": 1137,
    "g2e4": 1138,
    "g2h3": 1139,
    "g2g3": 1140,
    "g2f3": 1141,
    "g2h2": 1142,
    "g2f2": 1143,
    "g2e2": 1144,
    "g2d2": 1145,
    "g2c2": 1146,
    "g2b2": 1147,
    "g2a2": 1148,
    "g2h1": 1149,
    "g2g1": 1150,
    "g2f1": 1151,
    "g3g8": 1152,
    "g3b8": 1153,
    "g3g7": 1154,
    "g3c7": 1155,
    "g3g6": 1156,
    "g3d6": 1157,
    "g3g5": 1158,
    "g3e5": 1159,
    "g3h4": 1160,
    "g3g4": 1161,
    "g3f4": 1162,
    "g3h3": 1163,
    "g3f3": 1164,
    "g3e3": 1165,
    "g3d3": 1166,
    "g3c3": 1167,
    "g3b3": 1168,
    "g3a3": 1169,
    "g3h2": 1170,
    "g3g2": 1171,
    "g3f2": 1172,
    "g3g1": 1173,
    "g3e1": 1174,
    "g4g8": 1175,
    "g4c8": 1176,
    "g4g7": 1177,
    "g4d7": 1178,
    "g4g6": 1179,
    "g4e6": 1180,
    "g4h5": 1181,
    "g4g5": 1182,
    "g4f5": 1183,
    "g4h4": 1184,
    "g4f4": 1185,
    "g4e4": 1186,
    "g4d4": 1187,
    "g4c4": 1188,
    "g4b4": 1189,
    "g4a4": 1190,
    "g4h3": 1191,
    "g4g3": 1192,
    "g4f3": 1193,
    "g4g2": 1194,
    "g4e2": 1195,
    "g4g1": 1196,
    "g4d1": 1197,
    "g5g8": 1198,
    "g5d8": 1199,
    "g5g7": 1200,
    "g5e7": 1201,
    "g5h6": 1202,
    "g5g6": 1203,
    "g5f6": 1204,
    "g5h5": 1205,
    "g5f5": 1206,
    "g5e5": 1207,
    "g5d5": 1208,
    "g5c5": 1209,
    "g5b5": 1210,
    "g5a5": 1211,
    "g5h4": 1212,
    "g5g4": 1213,
    "g5f4": 1214,
    "g5g3": 1215,
    "g5e3": 1216,
    "g5g2": 1217,
    "g5d2": 1218,
    "g5g1": 1219,
    "g5c1": 1220,
    "g6g8": 1221,
    "g6e8": 1222,
    "g6h7": 1223,
    "g6g7": 1224,
    "g6f7": 1225,
    "g6h6": 1226,
    "g6f6": 1227,
    "g6e6": 1228,
    "g6d6": 1229,
    "g6c6": 1230,
    "g6b6": 1231,
    "g6a6": 1232,
    "g6h5": 1233,
    "g6g5": 1234,
    "g6f5": 1235,
    "g6g4": 1236,
    "g6e4": 1237,
    "g6g3": 1238,
    "g6d3": 1239,
    "g6g2": 1240,
    "g6c2": 1241,
    "g6g1": 1242,
    "g6b1": 1243,
    "g7h8": 1244,
    "g7g8": 1245,
    "g7f8": 1246,
    "g7h7": 1247,
    "g7f7": 1248,
    "g7e7": 1249,
    "g7d7": 1250,
    "g7c7": 1251,
    "g7b7": 1252,
    "g7a7": 1253,
    "g7h6": 1254,
    "g7g6": 1255,
    "g7f6": 1256,
    "g7g5": 1257,
    "g7e5": 1258,
    "g7g4": 1259,
    "g7d4": 1260,
    "g7g3": 1261,
    "g7c3": 1262,
    "g7g2": 1263,
    "g7b2": 1264,
    "g7g1": 1265,
    "g7a1": 1266,
    "g8h8": 1267,
    "g8f8": 1268,
    "g8e8": 1269,
    "g8d8": 1270,
    "g8c8": 1271,
    "g8b8": 1272,
    "g8a8": 1273,
    "g8h7": 1274,
    "g8g7": 1275,
    "g8f7": 1276,
    "g8g6": 1277,
    "g8e6": 1278,
    "g8g5": 1279,
    "g8d5": 1280,
    "g8g4": 1281,
    "g8c4": 1282,
    "g8g3": 1283,
    "g8b3": 1284,
    "g8g2": 1285,
    "g8a2": 1286,
    "g8g1": 1287,
    "h1h8": 1288,
    "h1a8": 1289,
    "h1h7": 1290,
    "h1b7": 1291,
    "h1h6": 1292,
    "h1c6": 1293,
    "h1h5": 1294,
    "h1d5": 1295,
    "h1h4": 1296,
    "h1e4": 1297,
    "h1h3": 1298,
    "h1f3": 1299,
    "h1h2": 1300,
    "h1g2": 1301,
    "h1g1": 1302,
    "h1f1": 1303,
    "h1e1": 1304,
    "h1d1": 1305,
    "h1c1": 1306,
    "h1b1": 1307,
    "h1a1": 1308,
    "h2h8": 1309,
    "h2b8": 1310,
    "h2h7": 1311,
    "h2c7": 1312,
    "h2h6": 1313,
    "h2d6": 1314,
    "h2h5": 1315,
    "h2e5": 1316,
    "h2h4": 1317,
    "h2f4": 1318,
    "h2h3": 1319,
    "h2g3": 1320,
    "h2g2": 1321,
    "h2f2": 1322,
    "h2e2": 1323,
    "h2d2": 1324,
    "h2c2": 1325,
    "h2b2": 1326,
    "h2a2": 1327,
    "h2h1": 1328,
    "h2g1": 1329,
    "h3h8": 1330,
    "h3c8": 1331,
    "h3h7": 1332,
    "h3d7": 1333,
    "h3h6": 1334,
    "h3e6": 1335,
    "h3h5": 1336,
    "h3f5": 1337,
    "h3h4": 1338,
    "h3g4": 1339,
    "h3g3": 1340,
    "h3f3": 1341,
    "h3e3": 1342,
    "h3d3": 1343,
    "h3c3": 1344,
    "h3b3": 1345,
    "h3a3": 1346,
    "h3h2": 1347,
    "h3g2": 1348,
    "h3h1": 1349,
    "h3f1": 1350,
    "h4h8": 1351,
    "h4d8": 1352,
    "h4h7": 1353,
    "h4e7": 1354,
    "h4h6": 1355,
    "h4f6": 1356,
    "h4h5": 1357,
    "h4g5": 1358,
    "h4g4": 1359,
    "h4f4": 1360,
    "h4e4": 1361,
    "h4d4": 1362,
    "h4c4": 1363,
    "h4b4": 1364,
    "h4a4": 1365,
    "h4h3": 1366,
    "h4g3": 1367,
    "h4h2": 1368,
    "h4f2": 1369,
    "h4h1": 1370,
    "h4e1": 1371,
    "h5h8": 1372,
    "h5e8": 1373,
    "h5h7": 1374,
    "h5f7": 1375,
    "h5h6": 1376,
    "h5g6": 1377,
    "h5g5": 1378,
    "h5f5": 1379,
    "h5e5": 1380,
    "h5d5": 1381,
    "h5c5": 1382,
    "h5b5": 1383,
    "h5a5": 1384,
    "h5h4": 1385,
    "h5g4": 1386,
    "h5h3": 1387,
    "h5f3": 1388,
    "h5h2": 1389,
    "h5e2": 1390,
    "h5h1": 1391,
    "h5d1": 1392,
    "h6h8": 1393,
    "h6f8": 1394,
    "h6h7": 1395,
    "h6g7": 1396,
    "h6g6": 1397,
    "h6f6": 1398,
    "h6e6": 1399,
    "h6d6": 1400,
    "h6c6": 1401,
    "h6b6": 1402,
    "h6a6": 1403,
    "h6h5": 1404,
    "h6g5": 1405,
    "h6h4": 1406,
    "h6f4": 1407,
    "h6h3": 1408,
    "h6e3": 1409,
    "h6h2": 1410,
    "h6d2": 1411,
    "h6h1": 1412,
    "h6c1": 1413,
    "h7h8": 1414,
    "h7g8": 1415,
    "h7g7": 1416,
    "h7f7": 1417,
    "h7e7": 1418,
    "h7d7": 1419,
    "h7c7": 1420,
    "h7b7": 1421,
    "h7a7": 1422,
    "h7h6": 1423,
    "h7g6": 1424,
    "h7h5": 1425,
    "h7f5": 1426,
    "h7h4": 1427,
    "h7e4": 1428,
    "h7h3": 1429,
    "h7d3": 1430,
    "h7h2": 1431,
    "h7c2": 1432,
    "h7h1": 1433,
    "h7b1": 1434,
    "h8g8": 1435,
    "h8f8": 1436,
    "h8e8": 1437,
    "h8d8": 1438,
    "h8c8": 1439,
    "h8b8": 1440,
    "h8a8": 1441,
    "h8h7": 1442,
    "h8g7": 1443,
    "h8h6": 1444,
    "h8f6": 1445,
    "h8h5": 1446,
    "h8e5": 1447,
    "h8h4": 1448,
    "h8d4": 1449,
    "h8h3": 1450,
    "h8c3": 1451,
    "h8h2": 1452,
    "h8b2": 1453,
    "h8h1": 1454,
    "h8a1": 1455,
    "a1b3": 1456,
    "a1c2": 1457,
    "a2b4": 1458,
    "a2c3": 1459,
    "a2c1": 1460,
    "a3b5": 1461,
    "a3c4": 1462,
    "a3c2": 1463,
    "a3b1": 1464,
    "a4b6": 1465,
    "a4c5": 1466,
    "a4c3": 1467,
    "a4b2": 1468,
    "a5b7": 1469,
    "a5c6": 1470,
    "a5c4": 1471,
    "a5b3": 1472,
    "a6b8": 1473,
    "a6c7": 1474,
    "a6c5": 1475,
    "a6b4": 1476,
    "a7c8": 1477,
    "a7c6": 1478,
    "a7b5": 1479,
    "a8c7": 1480,
    "a8b6": 1481,
    "b1c3": 1482,
    "b1a3": 1483,
    "b1d2": 1484,
    "b2c4": 1485,
    "b2a4": 1486,
    "b2d3": 1487,
    "b2d1": 1488,
    "b3c5": 1489,
    "b3a5": 1490,
    "b3d4": 1491,
    "b3d2": 1492,
    "b3c1": 1493,
    "b3a1": 1494,
    "b4c6": 1495,
    "b4a6": 1496,
    "b4d5": 1497,
    "b4d3": 1498,
    "b4c2": 1499,
    "b4a2": 1500,
    "b5c7": 1501,
    "b5a7": 1502,
    "b5d6": 1503,
    "b5d4": 1504,
    "b5c3": 1505,
    "b5a3": 1506,
    "b6c8": 1507,
    "b6a8": 1508,
    "b6d7": 1509,
    "b6d5": 1510,
    "b6c4": 1511,
    "b6a4": 1512,
    "b7d8": 1513,
    "b7d6": 1514,
    "b7c5": 1515,
    "b7a5": 1516,
    "b8d7": 1517,
    "b8c6": 1518,
    "b8a6": 1519,
    "c1d3": 1520,
    "c1b3": 1521,
    "c1e2": 1522,
    "c1a2": 1523,
    "c2d4": 1524,
    "c2b4": 1525,
    "c2e3": 1526,
    "c2a3": 1527,
    "c2e1": 1528,
    "c2a1": 1529,
    "c3d5": 1530,
    "c3b5": 1531,
    "c3e4": 1532,
    "c3a4": 1533,
    "c3e2": 1534,
    "c3a2": 1535,
    "c3d1": 1536,
    "c3b1": 1537,
    "c4d6": 1538,
    "c4b6": 1539,
    "c4e5": 1540,
    "c4a5": 1541,
    "c4e3": 1542,
    "c4a3": 1543,
    "c4d2": 1544,
    "c4b2": 1545,
    "c5d7": 1546,
    "c5b7": 1547,
    "c5e6": 1548,
    "c5a6": 1549,
    "c5e4": 1550,
    "c5a4": 1551,
    "c5d3": 1552,
    "c5b3": 1553,
    "c6d8": 1554,
    "c6b8": 1555,
    "c6e7": 1556,
    "c6a7": 1557,
    "c6e5": 1558,
    "c6a5": 1559,
    "c6d4": 1560,
    "c6b4": 1561,
    "c7e8": 1562,
    "c7a8": 1563,
    "c7e6": 1564,
    "c7a6": 1565,
    "c7d5": 1566,
    "c7b5": 1567,
    "c8e7": 1568,
    "c8a7": 1569,
    "c8d6": 1570,
    "c8b6": 1571,
    "d1e3": 1572,
    "d1c3": 1573,
    "d1f2": 1574,
    "d1b2": 1575,
    "d2e4": 1576,
    "d2c4": 1577,
    "d2f3": 1578,
    "d2b3": 1579,
    "d2f1": 1580,
    "d2b1": 1581,
    "d3e5": 1582,
    "d3c5": 1583,
    "d3f4": 1584,
    "d3b4": 1585,
    "d3f2": 1586,
    "d3b2": 1587,
    "d3e1": 1588,
    "d3c1": 1589,
    "d4e6": 1590,
    "d4c6": 1591,
    "d4f5": 1592,
    "d4b5": 1593,
    "d4f3": 1594,
    "d4b3": 1595,
    "d4e2": 1596,
    "d4c2": 1597,
    "d5e7": 1598,
    "d5c7": 1599,
    "d5f6": 1600,
    "d5b6": 1601,
    "d5f4": 1602,
    "d5b4": 1603,
    "d5e3": 1604,
    "d5c3": 1605,
    "d6e8": 1606,
    "d6c8": 1607,
    "d6f7": 1608,
    "d6b7": 1609,
    "d6f5": 1610,
    "d6b5": 1611,
    "d6e4": 1612,
    "d6c4": 1613,
    "d7f8": 1614,
    "d7b8": 1615,
    "d7f6": 1616,
    "d7b6": 1617,
    "d7e5": 1618,
    "d7c5": 1619,
    "d8f7": 1620,
    "d8b7": 1621,
    "d8e6": 1622,
    "d8c6": 1623,
    "e1f3": 1624,
    "e1d3": 1625,
    "e1g2": 1626,
    "e1c2": 1627,
    "e2f4": 1628,
    "e2d4": 1629,
    "e2g3": 1630,
    "e2c3": 1631,
    "e2g1": 1632,
    "e2c1": 1633,
    "e3f5": 1634,
    "e3d5": 1635,
    "e3g4": 1636,
    "e3c4": 1637,
    "e3g2": 1638,
    "e3c2": 1639,
    "e3f1": 1640,
    "e3d1": 1641,
    "e4f6": 1642,
    "e4d6": 1643,
    "e4g5": 1644,
    "e4c5": 1645,
    "e4g3": 1646,
    "e4c3": 1647,
    "e4f2": 1648,
    "e4d2": 1649,
    "e5f7": 1650,
    "e5d7": 1651,
    "e5g6": 1652,
    "e5c6": 1653,
    "e5g4": 1654,
    "e5c4": 1655,
    "e5f3": 1656,
    "e5d3": 1657,
    "e6f8": 1658,
    "e6d8": 1659,
    "e6g7": 1660,
    "e6c7": 1661,
    "e6g5": 1662,
    "e6c5": 1663,
    "e6f4": 1664,
    "e6d4": 1665,
    "e7g8": 1666,
    "e7c8": 1667,
    "e7g6": 1668,
    "e7c6": 1669,
    "e7f5": 1670,
    "e7d5": 1671,
    "e8g7": 1672,
    "e8c7": 1673,
    "e8f6": 1674,
    "e8d6": 1675,
    "f1g3": 1676,
    "f1e3": 1677,
    "f1h2": 1678,
    "f1d2": 1679,
    "f2g4": 1680,
    "f2e4": 1681,
    "f2h3": 1682,
    "f2d3": 1683,
    "f2h1": 1684,
    "f2d1": 1685,
    "f3g5": 1686,
    "f3e5": 1687,
    "f3h4": 1688,
    "f3d4": 1689,
    "f3h2": 1690,
    "f3d2": 1691,
    "f3g1": 1692,
    "f3e1": 1693,
    "f4g6": 1694,
    "f4e6": 1695,
    "f4h5": 1696,
    "f4d5": 1697,
    "f4h3": 1698,
    "f4d3": 1699,
    "f4g2": 1700,
    "f4e2": 1701,
    "f5g7": 1702,
    "f5e7": 1703,
    "f5h6": 1704,
    "f5d6": 1705,
    "f5h4": 1706,
    "f5d4": 1707,
    "f5g3": 1708,
    "f5e3": 1709,
    "f6g8": 1710,
    "f6e8": 1711,
    "f6h7": 1712,
    "f6d7": 1713,
    "f6h5": 1714,
    "f6d5": 1715,
    "f6g4": 1716,
    "f6e4": 1717,
    "f7h8": 1718,
    "f7d8": 1719,
    "f7h6": 1720,
    "f7d6": 1721,
    "f7g5": 1722,
    "f7e5": 1723,
    "f8h7": 1724,
    "f8d7": 1725,
    "f8g6": 1726,
    "f8e6": 1727,
    "g1h3": 1728,
    "g1f3": 1729,
    "g1e2": 1730,
    "g2h4": 1731,
    "g2f4": 1732,
    "g2e3": 1733,
    "g2e1": 1734,
    "g3h5": 1735,
    "g3f5": 1736,
    "g3e4": 1737,
    "g3e2": 1738,
    "g3h1": 1739,
    "g3f1": 1740,
    "g4h6": 1741,
    "g4f6": 1742,
    "g4e5": 1743,
    "g4e3": 1744,
    "g4h2": 1745,
    "g4f2": 1746,
    "g5h7": 1747,
    "g5f7": 1748,
    "g5e6": 1749,
    "g5e4": 1750,
    "g5h3": 1751,
    "g5f3": 1752,
    "g6h8": 1753,
    "g6f8": 1754,
    "g6e7": 1755,
    "g6e5": 1756,
    "g6h4": 1757,
    "g6f4": 1758,
    "g7e8": 1759,
    "g7e6": 1760,
    "g7h5": 1761,
    "g7f5": 1762,
    "g8e7": 1763,
    "g8h6": 1764,
    "g8f6": 1765,
    "h1g3": 1766,
    "h1f2": 1767,
    "h2g4": 1768,
    "h2f3": 1769,
    "h2f1": 1770,
    "h3g5": 1771,
    "h3f4": 1772,
    "h3f2": 1773,
    "h3g1": 1774,
    "h4g6": 1775,
    "h4f5": 1776,
    "h4f3": 1777,
    "h4g2": 1778,
    "h5g7": 1779,
    "h5f6": 1780,
    "h5f4": 1781,
    "h5g3": 1782,
    "h6g8": 1783,
    "h6f7": 1784,
    "h6f5": 1785,
    "h6g4": 1786,
    "h7f8": 1787,
    "h7f6": 1788,
    "h7g5": 1789,
    "h8f7": 1790,
    "h8g6": 1791,
    "a7b8q": 1792,
    "a7b8r": 1793,
    "a7b8b": 1794,
    "a7b8n": 1795,
    "a7a8q": 1796,
    "a7a8r": 1797,
    "a7a8b": 1798,
    "a7a8n": 1799,
    "a2b1q": 1800,
    "a2b1r": 1801,
    "a2b1b": 1802,
    "a2b1n": 1803,
    "a2a1q": 1804,
    "a2a1r": 1805,
    "a2a1b": 1806,
    "a2a1n": 1807,
    "b7c8q": 1808,
    "b7c8r": 1809,
    "b7c8b": 1810,
    "b7c8n": 1811,
    "b7a8q": 1812,
    "b7a8r": 1813,
    "b7a8b": 1814,
    "b7a8n": 1815,
    "b7b8q": 1816,
    "b7b8r": 1817,
    "b7b8b": 1818,
    "b7b8n": 1819,
    "b2c1q": 1820,
    "b2c1r": 1821,
    "b2c1b": 1822,
    "b2c1n": 1823,
    "b2a1q": 1824,
    "b2a1r": 1825,
    "b2a1b": 1826,
    "b2a1n": 1827,
    "b2b1q": 1828,
    "b2b1r": 1829,
    "b2b1b": 1830,
    "b2b1n": 1831,
    "c7d8q": 1832,
    "c7d8r": 1833,
    "c7d8b": 1834,
    "c7d8n": 1835,
    "c7b8q": 1836,
    "c7b8r": 1837,
    "c7b8b": 1838,
    "c7b8n": 1839,
    "c7c8q": 1840,
    "c7c8r": 1841,
    "c7c8b": 1842,
    "c7c8n": 1843,
    "c2d1q": 1844,
    "c2d1r": 1845,
    "c2d1b": 1846,
    "c2d1n": 1847,
    "c2b1q": 1848,
    "c2b1r": 1849,
    "c2b1b": 1850,
    "c2b1n": 1851,
    "c2c1q": 1852,
    "c2c1r": 1853,
    "c2c1b": 1854,
    "c2c1n": 1855,
    "d7e8q": 1856,
    "d7e8r": 1857,
    "d7e8b": 1858,
    "d7e8n": 1859,
    "d7c8q": 1860,
    "d7c8r": 1861,
    "d7c8b": 1862,
    "d7c8n": 1863,
    "d7d8q": 1864,
    "d7d8r": 1865,
    "d7d8b": 1866,
    "d7d8n": 1867,
    "d2e1q": 1868,
    "d2e1r": 1869,
    "d2e1b": 1870,
    "d2e1n": 1871,
    "d2c1q": 1872,
    "d2c1r": 1873,
    "d2c1b": 1874,
    "d2c1n": 1875,
    "d2d1q": 1876,
    "d2d1r": 1877,
    "d2d1b": 1878,
    "d2d1n": 1879,
    "e7f8q": 1880,
    "e7f8r": 1881,
    "e7f8b": 1882,
    "e7f8n": 1883,
    "e7d8q": 1884,
    "e7d8r": 1885,
    "e7d8b": 1886,
    "e7d8n": 1887,
    "e7e8q": 1888,
    "e7e8r": 1889,
    "e7e8b": 1890,
    "e7e8n": 1891,
    "e2f1q": 1892,
    "e2f1r": 1893,
    "e2f1b": 1894,
    "e2f1n": 1895,
    "e2d1q": 1896,
    "e2d1r": 1897,
    "e2d1b": 1898,
    "e2d1n": 1899,
    "e2e1q": 1900,
    "e2e1r": 1901,
    "e2e1b": 1902,
    "e2e1n": 1903,
    "f7g8q": 1904,
    "f7g8r": 1905,
    "f7g8b": 1906,
    "f7g8n": 1907,
    "f7e8q": 1908,
    "f7e8r": 1909,
    "f7e8b": 1910,
    "f7e8n": 1911,
    "f7f8q": 1912,
    "f7f8r": 1913,
    "f7f8b": 1914,
    "f7f8n": 1915,
    "f2g1q": 1916,
    "f2g1r": 1917,
    "f2g1b": 1918,
    "f2g1n": 1919,
    "f2e1q": 1920,
    "f2e1r": 1921,
    "f2e1b": 1922,
    "f2e1n": 1923,
    "f2f1q": 1924,
    "f2f1r": 1925,
    "f2f1b": 1926,
    "f2f1n": 1927,
    "g7h8q": 1928,
    "g7h8r": 1929,
    "g7h8b": 1930,
    "g7h8n": 1931,
    "g7f8q": 1932,
    "g7f8r": 1933,
    "g7f8b": 1934,
    "g7f8n": 1935,
    "g7g8q": 1936,
    "g7g8r": 1937,
    "g7g8b": 1938,
    "g7g8n": 1939,
    "g2h1q": 1940,
    "g2h1r": 1941,
    "g2h1b": 1942,
    "g2h1n": 1943,
    "g2f1q": 1944,
    "g2f1r": 1945,
    "g2f1b": 1946,
    "g2f1n": 1947,
    "g2g1q": 1948,
    "g2g1r": 1949,
    "g2g1b": 1950,
    "g2g1n": 1951,
    "h7g8q": 1952,
    "h7g8r": 1953,
    "h7g8b": 1954,
    "h7g8n": 1955,
    "h7h8q": 1956,
    "h7h8r": 1957,
    "h7h8b": 1958,
    "h7h8n": 1959,
    "h2g1q": 1960,
    "h2g1r": 1961,
    "h2g1b": 1962,
    "h2g1n": 1963,
    "h2h1q": 1964,
    "h2h1r": 1965,
    "h2h1b": 1966,
    "h2h1n": 1967,
    "<move>": 1968,
    "<loss>": 1969,
    "<pad>": 1970,
}
BOOL = {False: 0, True: 1}