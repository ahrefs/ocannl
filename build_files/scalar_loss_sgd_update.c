#include <stdio.h>
#include <stdlib.h>
/* Global declarations. */
#define w1_grad ((float*)0x5593ece52d10)
#define b2_grad ((float*)0x5593ecdbf880)
#define b3 ((float*)0x5593ecdac910)
#define w3 ((float*)0x7f9ca40456f0)
#define w2 ((float*)0x5593ece527a0)
#define w3_grad ((float*)0x5593ecdbf830)
#define b3_grad ((float*)0x5593ecd666d0)
#define b2 ((float*)0x5593ecda9970)
#define b1 ((float*)0x5593ecda9a10)
#define b1_grad ((float*)0x5593ecda99c0)
#define w1 ((float*)0x5593ece62de0)
#define learning_rate ((float*)0x5593ecdd9cb0)
#define w2_grad ((float*)0x5593ecdd8f20)

void scalar_loss_sgd_update(const char* log_file_name, int i157, int i158) {
  FILE* log_file = fopen(log_file_name, "w");
  /* Debug initial parameter state. */
  fprintf(log_file, "int i157 = %d\n", i157);
  fprintf(log_file, "int i158 = %d\n", i158);
  /* Local declarations and initialization. */
  float sgd_delta_b3[1];
  float n195[16];
  float n180[1];
  float n173[16];
  float n188[256];
  float n175[16];
  float sgd_delta_b2[16];
  float n190[256];
  float sgd_delta_b1[16];
  float n164[1];
  float n178[1];
  float sgd_delta_w2[256];
  float n162[1];
  float n185[32];
  float sgd_delta_w1[32];
  float n170[16];
  float n168[16];
  float sgd_delta_w3[16];
  float n193[16];
  float n183[32];
  float n158[1];
  
  /* Main logic. */
  fprintf(log_file,
  "COMMENT: scalar_loss sgd update\n");
  
  fprintf(log_file,
  "COMMENT: b1 param sgd step\n");
  
  for (int i240 = 0; i240 <= 15; ++i240) {
    { float new_set_v = ((0.000200) * b1[i240]);
      fprintf(log_file, "# n170[i240] := (0.000200 * b1[i240]);\n");
      fprintf(log_file, "n170[%u] = %f = (0.0002 * b1[%u]{=%f})\n", i240, new_set_v, i240, b1[i240]);
      fflush(log_file); n170[i240] = new_set_v;
    } 
  }
  
  for (int i242 = 0; i242 <= 15; ++i242) {
    { float new_set_v = (b1_grad[i242] + n170[i242]);
      fprintf(log_file, "# sgd_delta_b1[i242] := (b1.grad[i242] + n170[i242]);\n");
      fprintf(log_file, "sgd_delta_b1[%u] = %f = (b1_grad[%u]{=%f} + n170[%u]{=%f})\n", i242, new_set_v, i242
             , b1_grad[i242], i242, n170[i242]);
      fflush(log_file); sgd_delta_b1[i242] = new_set_v;
    } 
  }
  
  { float new_set_v = (float)i158; fprintf(log_file, "# n158[0] := i158;\n");
    fprintf(log_file, "n158[%u] = %f = i158\n", 0, new_set_v); fflush(log_file); 
    n158[0] = new_set_v;
  }
  
  { float new_set_v = ((40.000000) - n158[0]); fprintf(log_file, "# n162[0] := (40.000000 - n158[0]);\n");
    fprintf(log_file, "n162[%u] = %f = (40. - n158[%u]{=%f})\n", 0, new_set_v, 0, n158[0]); fflush(log_file);
    n162[0] = new_set_v;
  }
  
  { float new_set_v = ((0.100000) * n162[0]); fprintf(log_file, "# n164[0] := (0.100000 * n162[0]);\n");
    fprintf(log_file, "n164[%u] = %f = (0.1 * n162[%u]{=%f})\n", 0, new_set_v, 0, n162[0]); fflush(log_file);
    n164[0] = new_set_v;
  }
  
  { float new_set_v = (n164[0] / (20.000000));
    fprintf(log_file, "# learning_rate[0] := (n164[0] / 20.000000);\n");
    fprintf(log_file, "learning_rate[%u] = %f = (n164[%u]{=%f} / 20.)\n", 0, new_set_v, 0, n164[0]);
    fflush(log_file); learning_rate[0] = new_set_v;
  }
  
  for (int i244 = 0; i244 <= 15; ++i244) {
    { float new_set_v = (learning_rate[0] * sgd_delta_b1[i244]);
      fprintf(log_file, "# n168[i244] := (learning_rate[0] * sgd_delta_b1[i244]);\n");
      fprintf(log_file, "n168[%u] = %f = (learning_rate[%u]{=%f} * sgd_delta_b1[%u]{=%f})\n", i244,
             new_set_v, 0, learning_rate[0], i244, sgd_delta_b1[i244]);
      fflush(log_file); n168[i244] = new_set_v;
    } 
  }
  
  for (int i246 = 0; i246 <= 15; ++i246) {
    { float new_set_v = (b1[i246] - n168[i246]);
      fprintf(log_file, "# b1[i246] := (b1[i246] - n168[i246]);\n");
      fprintf(log_file, "b1[%u] = %f = (b1[%u]{=%f} - n168[%u]{=%f})\n", i246, new_set_v, i246, b1[i246],
             i246, n168[i246]);
      fflush(log_file); b1[i246] = new_set_v;
    } 
  }
  
  fprintf(log_file,
  "COMMENT: end\n");
  
  fprintf(log_file,
  "COMMENT: b2 param sgd step\n");
  
  for (int i248 = 0; i248 <= 15; ++i248) {
    { float new_set_v = ((0.000200) * b2[i248]);
      fprintf(log_file, "# n175[i248] := (0.000200 * b2[i248]);\n");
      fprintf(log_file, "n175[%u] = %f = (0.0002 * b2[%u]{=%f})\n", i248, new_set_v, i248, b2[i248]);
      fflush(log_file); n175[i248] = new_set_v;
    } 
  }
  
  for (int i250 = 0; i250 <= 15; ++i250) {
    { float new_set_v = (b2_grad[i250] + n175[i250]);
      fprintf(log_file, "# sgd_delta_b2[i250] := (b2.grad[i250] + n175[i250]);\n");
      fprintf(log_file, "sgd_delta_b2[%u] = %f = (b2_grad[%u]{=%f} + n175[%u]{=%f})\n", i250, new_set_v, i250
             , b2_grad[i250], i250, n175[i250]);
      fflush(log_file); sgd_delta_b2[i250] = new_set_v;
    } 
  }
  
  for (int i252 = 0; i252 <= 15; ++i252) {
    { float new_set_v = (learning_rate[0] * sgd_delta_b2[i252]);
      fprintf(log_file, "# n173[i252] := (learning_rate[0] * sgd_delta_b2[i252]);\n");
      fprintf(log_file, "n173[%u] = %f = (learning_rate[%u]{=%f} * sgd_delta_b2[%u]{=%f})\n", i252,
             new_set_v, 0, learning_rate[0], i252, sgd_delta_b2[i252]);
      fflush(log_file); n173[i252] = new_set_v;
    } 
  }
  
  for (int i254 = 0; i254 <= 15; ++i254) {
    { float new_set_v = (b2[i254] - n173[i254]);
      fprintf(log_file, "# b2[i254] := (b2[i254] - n173[i254]);\n");
      fprintf(log_file, "b2[%u] = %f = (b2[%u]{=%f} - n173[%u]{=%f})\n", i254, new_set_v, i254, b2[i254],
             i254, n173[i254]);
      fflush(log_file); b2[i254] = new_set_v;
    } 
  }
  
  fprintf(log_file,
  "COMMENT: end\n");
  
  fprintf(log_file,
  "COMMENT: b3 param sgd step\n");
  
  { float new_set_v = ((0.000200) * b3[0]); fprintf(log_file, "# n180[0] := (0.000200 * b3[0]);\n");
    fprintf(log_file, "n180[%u] = %f = (0.0002 * b3[%u]{=%f})\n", 0, new_set_v, 0, b3[0]); fflush(log_file);
    n180[0] = new_set_v;
  }
  
  { float new_set_v = (b3_grad[0] + n180[0]);
    fprintf(log_file, "# sgd_delta_b3[0] := (b3.grad[0] + n180[0]);\n");
    fprintf(log_file, "sgd_delta_b3[%u] = %f = (b3_grad[%u]{=%f} + n180[%u]{=%f})\n", 0, new_set_v, 0,
           b3_grad[0], 0, n180[0]);
    fflush(log_file); sgd_delta_b3[0] = new_set_v;
  }
  
  { float new_set_v = (learning_rate[0] * sgd_delta_b3[0]);
    fprintf(log_file, "# n178[0] := (learning_rate[0] * sgd_delta_b3[0]);\n");
    fprintf(log_file, "n178[%u] = %f = (learning_rate[%u]{=%f} * sgd_delta_b3[%u]{=%f})\n", 0, new_set_v, 0,
           learning_rate[0], 0, sgd_delta_b3[0]);
    fflush(log_file); n178[0] = new_set_v;
  }
  
  { float new_set_v = (b3[0] - n178[0]); fprintf(log_file, "# b3[0] := (b3[0] - n178[0]);\n");
    fprintf(log_file, "b3[%u] = %f = (b3[%u]{=%f} - n178[%u]{=%f})\n", 0, new_set_v, 0, b3[0], 0, n178[0]);
    fflush(log_file); b3[0] = new_set_v;
  }
  
  fprintf(log_file,
  "COMMENT: end\n");
  
  fprintf(log_file,
  "COMMENT: w1 param sgd step\n");
  
  for (int i257 = 0; i257 <= 15; ++i257) {
    for (int i258 = 0; i258 <= 1; ++i258) {
      { float new_set_v = ((0.000200) * w1[i257 * 2 + i258]);
        fprintf(log_file, "# n185[i257, i258] := (0.000200 * w1[i257, i258]);\n");
        fprintf(log_file, "n185[%u] = %f = (0.0002 * w1[%u]{=%f})\n", i257 * 2 + i258, new_set_v,
               i257 * 2 + i258, w1[i257 * 2 + i258]);
        fflush(log_file); n185[i257 * 2 + i258] = new_set_v;
      } 
    }
  }
  
  for (int i261 = 0; i261 <= 15; ++i261) {
    for (int i262 = 0; i262 <= 1; ++i262) {
      { float new_set_v = (w1_grad[i261 * 2 + i262] + n185[i261 * 2 + i262]);
        fprintf(log_file, "# sgd_delta_w1[i261, i262] := (w1.grad[i261, i262] + n185[i261, i262]);\n");
        fprintf(log_file, "sgd_delta_w1[%u] = %f = (w1_grad[%u]{=%f} + n185[%u]{=%f})\n", i261 * 2 + i262,
               new_set_v, i261 * 2 + i262, w1_grad[i261 * 2 + i262], i261 * 2 + i262, n185[i261 * 2 + i262]);
        fflush(log_file); sgd_delta_w1[i261 * 2 + i262] = new_set_v;
      } 
    }
  }
  
  for (int i265 = 0; i265 <= 15; ++i265) {
    for (int i266 = 0; i266 <= 1; ++i266) {
      { float new_set_v = (learning_rate[0] * sgd_delta_w1[i265 * 2 + i266]);
        fprintf(log_file, "# n183[i265, i266] := (learning_rate[0] * sgd_delta_w1[i265, i266]);\n");
        fprintf(log_file, "n183[%u] = %f = (learning_rate[%u]{=%f} * sgd_delta_w1[%u]{=%f})\n",
               i265 * 2 + i266, new_set_v, 0, learning_rate[0], i265 * 2 + i266,
               sgd_delta_w1[i265 * 2 + i266]);
        fflush(log_file); n183[i265 * 2 + i266] = new_set_v;
      } 
    }
  }
  
  for (int i269 = 0; i269 <= 15; ++i269) {
    for (int i270 = 0; i270 <= 1; ++i270) {
      { float new_set_v = (w1[i269 * 2 + i270] - n183[i269 * 2 + i270]);
        fprintf(log_file, "# w1[i269, i270] := (w1[i269, i270] - n183[i269, i270]);\n");
        fprintf(log_file, "w1[%u] = %f = (w1[%u]{=%f} - n183[%u]{=%f})\n", i269 * 2 + i270, new_set_v,
               i269 * 2 + i270, w1[i269 * 2 + i270], i269 * 2 + i270, n183[i269 * 2 + i270]);
        fflush(log_file); w1[i269 * 2 + i270] = new_set_v;
      } 
    }
  }
  
  fprintf(log_file,
  "COMMENT: end\n");
  
  fprintf(log_file,
  "COMMENT: w2 param sgd step\n");
  
  for (int i273 = 0; i273 <= 15; ++i273) {
    for (int i274 = 0; i274 <= 15; ++i274) {
      { float new_set_v = ((0.000200) * w2[i273 * 16 + i274]);
        fprintf(log_file, "# n190[i273, i274] := (0.000200 * w2[i273, i274]);\n");
        fprintf(log_file, "n190[%u] = %f = (0.0002 * w2[%u]{=%f})\n", i273 * 16 + i274, new_set_v,
               i273 * 16 + i274, w2[i273 * 16 + i274]);
        fflush(log_file); n190[i273 * 16 + i274] = new_set_v;
      } 
    }
  }
  
  for (int i277 = 0; i277 <= 15; ++i277) {
    for (int i278 = 0; i278 <= 15; ++i278) {
      { float new_set_v = (w2_grad[i277 * 16 + i278] + n190[i277 * 16 + i278]);
        fprintf(log_file, "# sgd_delta_w2[i277, i278] := (w2.grad[i277, i278] + n190[i277, i278]);\n");
        fprintf(log_file, "sgd_delta_w2[%u] = %f = (w2_grad[%u]{=%f} + n190[%u]{=%f})\n", i277 * 16 + i278,
               new_set_v, i277 * 16 + i278, w2_grad[i277 * 16 + i278], i277 * 16 + i278,
               n190[i277 * 16 + i278]);
        fflush(log_file); sgd_delta_w2[i277 * 16 + i278] = new_set_v;
      } 
    }
  }
  
  for (int i281 = 0; i281 <= 15; ++i281) {
    for (int i282 = 0; i282 <= 15; ++i282) {
      { float new_set_v = (learning_rate[0] * sgd_delta_w2[i281 * 16 + i282]);
        fprintf(log_file, "# n188[i281, i282] := (learning_rate[0] * sgd_delta_w2[i281, i282]);\n");
        fprintf(log_file, "n188[%u] = %f = (learning_rate[%u]{=%f} * sgd_delta_w2[%u]{=%f})\n",
               i281 * 16 + i282, new_set_v, 0, learning_rate[0], i281 * 16 + i282,
               sgd_delta_w2[i281 * 16 + i282]);
        fflush(log_file); n188[i281 * 16 + i282] = new_set_v;
      } 
    }
  }
  
  for (int i285 = 0; i285 <= 15; ++i285) {
    for (int i286 = 0; i286 <= 15; ++i286) {
      { float new_set_v = (w2[i285 * 16 + i286] - n188[i285 * 16 + i286]);
        fprintf(log_file, "# w2[i285, i286] := (w2[i285, i286] - n188[i285, i286]);\n");
        fprintf(log_file, "w2[%u] = %f = (w2[%u]{=%f} - n188[%u]{=%f})\n", i285 * 16 + i286, new_set_v,
               i285 * 16 + i286, w2[i285 * 16 + i286], i285 * 16 + i286, n188[i285 * 16 + i286]);
        fflush(log_file); w2[i285 * 16 + i286] = new_set_v;
      } 
    }
  }
  
  fprintf(log_file,
  "COMMENT: end\n");
  
  fprintf(log_file,
  "COMMENT: w3 param sgd step\n");
  
  for (int i288 = 0; i288 <= 15; ++i288) {
    { float new_set_v = ((0.000200) * w3[0 * 16 + i288]);
      fprintf(log_file, "# n195[0, i288] := (0.000200 * w3[0, i288]);\n");
      fprintf(log_file, "n195[%u] = %f = (0.0002 * w3[%u]{=%f})\n", 0 * 16 + i288, new_set_v, 0 * 16 + i288,
             w3[0 * 16 + i288]);
      fflush(log_file); n195[0 * 16 + i288] = new_set_v;
    } 
  }
  
  for (int i290 = 0; i290 <= 15; ++i290) {
    { float new_set_v = (w3_grad[0 * 16 + i290] + n195[0 * 16 + i290]);
      fprintf(log_file, "# sgd_delta_w3[0, i290] := (w3.grad[0, i290] + n195[0, i290]);\n");
      fprintf(log_file, "sgd_delta_w3[%u] = %f = (w3_grad[%u]{=%f} + n195[%u]{=%f})\n", 0 * 16 + i290,
             new_set_v, 0 * 16 + i290, w3_grad[0 * 16 + i290], 0 * 16 + i290, n195[0 * 16 + i290]);
      fflush(log_file); sgd_delta_w3[0 * 16 + i290] = new_set_v;
    } 
  }
  
  for (int i292 = 0; i292 <= 15; ++i292) {
    { float new_set_v = (learning_rate[0] * sgd_delta_w3[0 * 16 + i292]);
      fprintf(log_file, "# n193[0, i292] := (learning_rate[0] * sgd_delta_w3[0, i292]);\n");
      fprintf(log_file, "n193[%u] = %f = (learning_rate[%u]{=%f} * sgd_delta_w3[%u]{=%f})\n", 0 * 16 + i292,
             new_set_v, 0, learning_rate[0], 0 * 16 + i292, sgd_delta_w3[0 * 16 + i292]);
      fflush(log_file); n193[0 * 16 + i292] = new_set_v;
    } 
  }
  
  for (int i294 = 0; i294 <= 15; ++i294) {
    { float new_set_v = (w3[0 * 16 + i294] - n193[0 * 16 + i294]);
      fprintf(log_file, "# w3[0, i294] := (w3[0, i294] - n193[0, i294]);\n");
      fprintf(log_file, "w3[%u] = %f = (w3[%u]{=%f} - n193[%u]{=%f})\n", 0 * 16 + i294, new_set_v,
             0 * 16 + i294, w3[0 * 16 + i294], 0 * 16 + i294, n193[0 * 16 + i294]);
      fflush(log_file); w3[0 * 16 + i294] = new_set_v;
    } 
  }
  
  fprintf(log_file,
  "COMMENT: end\n");
  
  fprintf(log_file,
  "COMMENT: end\n");
  
}

