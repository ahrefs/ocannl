#include <stdio.h>
#include <stdlib.h>
/* Global declarations. */
#define moons_flat ((float*)0x5593ece7c560)
#define moons_classes ((float*)0x5593ecdd5d30)

void scalar_loss_gradient_update(
    const char* log_file_name, float *b1, float *b1_grad, float *b2, float *b2_grad, float *b3,
    float *b3_grad, float *scalar_loss, float *w1, float *w1_grad, float *w2, float *w2_grad, float *w3,
    float *w3_grad, int i157, int i158
) {
  FILE* log_file = fopen(log_file_name, "w");
  /* Debug initial parameter state. */
  fprintf(log_file, "float *b1 = %p\n", (void*)b1);
  fprintf(log_file, "float *b1_grad = %p\n", (void*)b1_grad);
  fprintf(log_file, "float *b2 = %p\n", (void*)b2);
  fprintf(log_file, "float *b2_grad = %p\n", (void*)b2_grad);
  fprintf(log_file, "float *b3 = %p\n", (void*)b3);
  fprintf(log_file, "float *b3_grad = %p\n", (void*)b3_grad);
  fprintf(log_file, "float *scalar_loss = %p\n", (void*)scalar_loss);
  fprintf(log_file, "float *w1 = %p\n", (void*)w1);
  fprintf(log_file, "float *w1_grad = %p\n", (void*)w1_grad);
  fprintf(log_file, "float *w2 = %p\n", (void*)w2);
  fprintf(log_file, "float *w2_grad = %p\n", (void*)w2_grad);
  fprintf(log_file, "float *w3 = %p\n", (void*)w3);
  fprintf(log_file, "float *w3_grad = %p\n", (void*)w3_grad);
  fprintf(log_file, "int i157 = %d\n", i157);
  fprintf(log_file, "int i158 = %d\n", i158);
  /* Local declarations and initialization. */
  memset(w1_grad, 0, 128);
  float n123[1920] = {0};
  float expectation[120];
  memset(b2_grad, 0, 64);
  float n133[1920];
  float mlp[120];
  float n154[1];
  float n135_grad[120] = {0};
  float n135[120] = {0};
  float n144_grad[120] = {0};
  float input[240];
  memset(w3_grad, 0, 64);
  float n129_grad[1920] = {0};
  float n125[1920];
  float mlp_grad[120] = {0};
  memset(b3_grad, 0, 4);
  float n125_grad[1920] = {0};
  memset(b1_grad, 0, 64);
  float n139[120];
  float n142_grad[120] = {0};
  float n127_grad[1920] = {0};
  float n123_grad[1920] = {0};
  float n129[1920] = {0};
  float n147[1] = {0};
  float n131[1920];
  float n144[120];
  float n127[1920];
  float n131_grad[1920] = {0};
  float n142[120];
  memset(w2_grad, 0, 1024);
  float n139_grad[120] = {0};
  float n133_grad[1920] = {0};
  
  /* Main logic. */
  fprintf(log_file,
  "COMMENT: scalar_loss gradient update\n");
  
  fprintf(log_file,
  "COMMENT: scalar_loss fwd\n");
  
  for (int i160 = 0; i160 <= 119; ++i160) {
    { float new_set_v = moons_classes[(i157 * 120 + i160) * 1 + 0];
      fprintf(log_file, "# expectation[i160, 0] := moons_classes[i157, i160, 0];\n");
      fprintf(log_file, "expectation[%u] = %f = moons_classes[%u]{=%f}\n", i160 * 1 + 0, new_set_v,
             (i157 * 120 + i160) * 1 + 0, moons_classes[(i157 * 120 + i160) * 1 + 0]);
      fflush(log_file); expectation[i160 * 1 + 0] = new_set_v;
    } 
  }
  
  for (int i163 = 0; i163 <= 119; ++i163) {
    for (int i164 = 0; i164 <= 1; ++i164) {
      { float new_set_v = moons_flat[(i157 * 120 + i163) * 2 + i164];
        fprintf(log_file, "# input[i163, i164] := moons_flat[i157, i163, i164];\n");
        fprintf(log_file, "input[%u] = %f = moons_flat[%u]{=%f}\n", i163 * 2 + i164, new_set_v,
               (i157 * 120 + i163) * 2 + i164, moons_flat[(i157 * 120 + i163) * 2 + i164]);
        fflush(log_file); input[i163 * 2 + i164] = new_set_v;
      } 
    }
  }
  
  
  for (int i168 = 0; i168 <= 119; ++i168) {
    for (int i169 = 0; i169 <= 15; ++i169) {
      for (int i170 = 0; i170 <= 1; ++i170) {
        { float new_set_v = (n123[i168 * 16 + i169] + (w1[i169 * 2 + i170] * input[i168 * 2 + i170]));
          fprintf(log_file,
                 "# n123[i168, i169] :=$  (n123[i168, i169] + (w1[i169, i170] * input[i168, i170]));\n");
          fprintf(log_file, "n123[%u] = %f = (n123[%u]{=%f} + (w1[%u]{=%f} * input[%u]{=%f}))\n",
                 i168 * 16 + i169, new_set_v, i168 * 16 + i169, n123[i168 * 16 + i169], i169 * 2 + i170,
                 w1[i169 * 2 + i170], i168 * 2 + i170, input[i168 * 2 + i170]);
          fflush(log_file); n123[i168 * 16 + i169] = new_set_v;
        } 
      }
    }
  }
  
  for (int i173 = 0; i173 <= 119; ++i173) {
    for (int i174 = 0; i174 <= 15; ++i174) {
      { float new_set_v = (b1[i174] + n123[i173 * 16 + i174]);
        fprintf(log_file, "# n125[i173, i174] := (b1[i174] + n123[i173, i174]);\n");
        fprintf(log_file, "n125[%u] = %f = (b1[%u]{=%f} + n123[%u]{=%f})\n", i173 * 16 + i174, new_set_v,
               i174, b1[i174], i173 * 16 + i174, n123[i173 * 16 + i174]);
        fflush(log_file); n125[i173 * 16 + i174] = new_set_v;
      } 
    }
  }
  
  for (int i177 = 0; i177 <= 119; ++i177) {
    for (int i178 = 0; i178 <= 15; ++i178) {
      { float new_set_v = (n125[i177 * 16 + i178] > 0.0 ? n125[i177 * 16 + i178] : 0.0);
        fprintf(log_file, "# n127[i177, i178] := relu(n125[i177, i178]);\n");
        fprintf(log_file, "n127[%u] = %f = (n125[%u]{=%f} > 0.0 ? n125[%u]{=%f} : 0.0)\n", i177 * 16 + i178,
               new_set_v, i177 * 16 + i178, n125[i177 * 16 + i178], i177 * 16 + i178,
               n125[i177 * 16 + i178]);
        fflush(log_file); n127[i177 * 16 + i178] = new_set_v;
      } 
    }
  }
  
  
  for (int i182 = 0; i182 <= 119; ++i182) {
    for (int i183 = 0; i183 <= 15; ++i183) {
      for (int i184 = 0; i184 <= 15; ++i184) {
        { float new_set_v = (n129[i182 * 16 + i183] + (w2[i183 * 16 + i184] * n127[i182 * 16 + i184]));
          fprintf(log_file,
                 "# n129[i182, i183] := (n129[i182, i183] + (w2[i183, i184] * n127[i182, i184]));\n");
          fprintf(log_file, "n129[%u] = %f = (n129[%u]{=%f} + (w2[%u]{=%f} * n127[%u]{=%f}))\n",
                 i182 * 16 + i183, new_set_v, i182 * 16 + i183, n129[i182 * 16 + i183], i183 * 16 + i184,
                 w2[i183 * 16 + i184], i182 * 16 + i184, n127[i182 * 16 + i184]);
          fflush(log_file); n129[i182 * 16 + i183] = new_set_v;
        } 
      }
    }
  }
  
  for (int i187 = 0; i187 <= 119; ++i187) {
    for (int i188 = 0; i188 <= 15; ++i188) {
      { float new_set_v = (b2[i188] + n129[i187 * 16 + i188]);
        fprintf(log_file, "# n131[i187, i188] := (b2[i188] + n129[i187, i188]);\n");
        fprintf(log_file, "n131[%u] = %f = (b2[%u]{=%f} + n129[%u]{=%f})\n", i187 * 16 + i188, new_set_v,
               i188, b2[i188], i187 * 16 + i188, n129[i187 * 16 + i188]);
        fflush(log_file); n131[i187 * 16 + i188] = new_set_v;
      } 
    }
  }
  
  for (int i191 = 0; i191 <= 119; ++i191) {
    for (int i192 = 0; i192 <= 15; ++i192) {
      { float new_set_v = (n131[i191 * 16 + i192] > 0.0 ? n131[i191 * 16 + i192] : 0.0);
        fprintf(log_file, "# n133[i191, i192] := relu(n131[i191, i192]);\n");
        fprintf(log_file, "n133[%u] = %f = (n131[%u]{=%f} > 0.0 ? n131[%u]{=%f} : 0.0)\n", i191 * 16 + i192,
               new_set_v, i191 * 16 + i192, n131[i191 * 16 + i192], i191 * 16 + i192,
               n131[i191 * 16 + i192]);
        fflush(log_file); n133[i191 * 16 + i192] = new_set_v;
      } 
    }
  }
  
  
  for (int i195 = 0; i195 <= 119; ++i195) {
    for (int i196 = 0; i196 <= 15; ++i196) {
      { float new_set_v = (n135[i195 * 1 + 0] + (w3[0 * 16 + i196] * n133[i195 * 16 + i196]));
        fprintf(log_file, "# n135[i195, 0] := (n135[i195, 0] + (w3[0, i196] * n133[i195, i196]));\n");
        fprintf(log_file, "n135[%u] = %f = (n135[%u]{=%f} + (w3[%u]{=%f} * n133[%u]{=%f}))\n", i195 * 1 + 0,
               new_set_v, i195 * 1 + 0, n135[i195 * 1 + 0], 0 * 16 + i196, w3[0 * 16 + i196],
               i195 * 16 + i196, n133[i195 * 16 + i196]);
        fflush(log_file); n135[i195 * 1 + 0] = new_set_v;
      } 
    }
  }
  
  for (int i198 = 0; i198 <= 119; ++i198) {
    { float new_set_v = (b3[0] + n135[i198 * 1 + 0]);
      fprintf(log_file, "# mlp[i198, 0] := (b3[0] + n135[i198, 0]);\n");
      fprintf(log_file, "mlp[%u] = %f = (b3[%u]{=%f} + n135[%u]{=%f})\n", i198 * 1 + 0, new_set_v, 0, b3[0],
             i198 * 1 + 0, n135[i198 * 1 + 0]);
      fflush(log_file); mlp[i198 * 1 + 0] = new_set_v;
    } 
  }
  
  for (int i200 = 0; i200 <= 119; ++i200) {
    { float new_set_v = (expectation[i200 * 1 + 0] * mlp[i200 * 1 + 0]);
      fprintf(log_file, "# n139[i200, 0] := (expectation[i200, 0] * mlp[i200, 0]);\n");
      fprintf(log_file, "n139[%u] = %f = (expectation[%u]{=%f} * mlp[%u]{=%f})\n", i200 * 1 + 0, new_set_v,
             i200 * 1 + 0, expectation[i200 * 1 + 0], i200 * 1 + 0, mlp[i200 * 1 + 0]);
      fflush(log_file); n139[i200 * 1 + 0] = new_set_v;
    } 
  }
  
  for (int i202 = 0; i202 <= 119; ++i202) {
    { float new_set_v = ((1.000000) - n139[i202 * 1 + 0]);
      fprintf(log_file, "# n142[i202, 0] := (1.000000 - n139[i202, 0]);\n");
      fprintf(log_file, "n142[%u] = %f = (1. - n139[%u]{=%f})\n", i202 * 1 + 0, new_set_v, i202 * 1 + 0,
             n139[i202 * 1 + 0]);
      fflush(log_file); n142[i202 * 1 + 0] = new_set_v;
    } 
  }
  
  for (int i204 = 0; i204 <= 119; ++i204) {
    { float new_set_v = (n142[i204 * 1 + 0] > 0.0 ? n142[i204 * 1 + 0] : 0.0);
      fprintf(log_file, "# n144[i204, 0] := relu(n142[i204, 0]);\n");
      fprintf(log_file, "n144[%u] = %f = (n142[%u]{=%f} > 0.0 ? n142[%u]{=%f} : 0.0)\n", i204 * 1 + 0,
             new_set_v, i204 * 1 + 0, n142[i204 * 1 + 0], i204 * 1 + 0, n142[i204 * 1 + 0]);
      fflush(log_file); n144[i204 * 1 + 0] = new_set_v;
    } 
  }
  
  
  for (int i207 = 0; i207 <= 119; ++i207) {
    { float new_set_v = (n147[0] + n144[i207 * 1 + 0]);
      fprintf(log_file, "# n147[0] := (n147[0] + n144[i207, 0]);\n");
      fprintf(log_file, "n147[%u] = %f = (n147[%u]{=%f} + n144[%u]{=%f})\n", 0, new_set_v, 0, n147[0],
             i207 * 1 + 0, n144[i207 * 1 + 0]);
      fflush(log_file); n147[0] = new_set_v;
    } 
  }
  
  { float new_set_v = (n147[0] / (120.000000));
    fprintf(log_file, "# scalar_loss[0] := (n147[0] / 120.000000);\n");
    fprintf(log_file, "scalar_loss[%u] = %f = (n147[%u]{=%f} / 120.)\n", 0, new_set_v, 0, n147[0]);
    fflush(log_file); scalar_loss[0] = new_set_v;
  }
  
  fprintf(log_file,
  "COMMENT: end\n");
  
  fprintf(log_file,
  "COMMENT: scalar_loss zero grads\n");
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  fprintf(log_file,
  "COMMENT: end\n");
  
  fprintf(log_file,
  "COMMENT: scalar_loss bprop\n");
  
  { float new_set_v = ((-1.000000) * n147[0]); fprintf(log_file, "# n154[0] := (-1.000000 * n147[0]);\n");
    fprintf(log_file, "n154[%u] = %f = (-1. * n147[%u]{=%f})\n", 0, new_set_v, 0, n147[0]); fflush(log_file);
    n154[0] = new_set_v;
  }
  
  for (int i208 = 0; i208 <= 119; ++i208) {
    { float new_set_v = (n144_grad[i208 * 1 + 0] + (0.008333));
      fprintf(log_file, "# n144.grad[i208, 0] := (n144.grad[i208, 0] + 0.008333);\n");
      fprintf(log_file, "n144_grad[%u] = %f = (n144_grad[%u]{=%f} + 0.0083333333333333332)\n", i208 * 1 + 0,
             new_set_v, i208 * 1 + 0, n144_grad[i208 * 1 + 0]);
      fflush(log_file); n144_grad[i208 * 1 + 0] = new_set_v;
    } 
  }
  
  for (int i209 = 0; i209 <= 119; ++i209) {
    {
      float new_set_v =
        (n142_grad[i209 * 1 + 0] + (n144[i209 * 1 + 0] > 0.0 ? n144_grad[i209 * 1 + 0] : 0.0));
      fprintf(log_file,
             "# n142.grad[i209, 0] :=$  (n142.grad[i209, 0] + (n144[i209, 0] > 0.0 ? n144.grad[i209, 0] : 0.0));\n");
      fprintf(log_file,
             "n142_grad[%u] = %f = (n142_grad[%u]{=%f} + (n144[%u]{=%f} > 0.0 ? n144_grad[%u]{=%f} : 0.0))\n",
             i209 * 1 + 0, new_set_v, i209 * 1 + 0, n142_grad[i209 * 1 + 0], i209 * 1 + 0, n144[i209 * 1 + 0]
             , i209 * 1 + 0, n144_grad[i209 * 1 + 0]);
      fflush(log_file); n142_grad[i209 * 1 + 0] = new_set_v;
    } 
  }
  
  for (int i210 = 0; i210 <= 119; ++i210) {
    { float new_set_v = (n139_grad[i210 * 1 + 0] - n142_grad[i210 * 1 + 0]);
      fprintf(log_file, "# n139.grad[i210, 0] := (n139.grad[i210, 0] - n142.grad[i210, 0]);\n");
      fprintf(log_file, "n139_grad[%u] = %f = (n139_grad[%u]{=%f} - n142_grad[%u]{=%f})\n", i210 * 1 + 0,
             new_set_v, i210 * 1 + 0, n139_grad[i210 * 1 + 0], i210 * 1 + 0, n142_grad[i210 * 1 + 0]);
      fflush(log_file); n139_grad[i210 * 1 + 0] = new_set_v;
    } 
  }
  
  for (int i211 = 0; i211 <= 119; ++i211) {
    { float new_set_v = (mlp_grad[i211 * 1 + 0] + (expectation[i211 * 1 + 0] * n139_grad[i211 * 1 + 0]));
      fprintf(log_file,
             "# mlp.grad[i211, 0] :=$  (mlp.grad[i211, 0] + (expectation[i211, 0] * n139.grad[i211, 0]));\n");
      fprintf(log_file,
             "mlp_grad[%u] = %f = (mlp_grad[%u]{=%f} + (expectation[%u]{=%f} * n139_grad[%u]{=%f}))\n",
             i211 * 1 + 0, new_set_v, i211 * 1 + 0, mlp_grad[i211 * 1 + 0], i211 * 1 + 0,
             expectation[i211 * 1 + 0], i211 * 1 + 0, n139_grad[i211 * 1 + 0]);
      fflush(log_file); mlp_grad[i211 * 1 + 0] = new_set_v;
    } 
  }
  
  for (int i212 = 0; i212 <= 119; ++i212) {
    { float new_set_v = (b3_grad[0] + mlp_grad[i212 * 1 + 0]);
      fprintf(log_file, "# b3.grad[0] := (b3.grad[0] + mlp.grad[i212, 0]);\n");
      fprintf(log_file, "b3_grad[%u] = %f = (b3_grad[%u]{=%f} + mlp_grad[%u]{=%f})\n", 0, new_set_v, 0,
             b3_grad[0], i212 * 1 + 0, mlp_grad[i212 * 1 + 0]);
      fflush(log_file); b3_grad[0] = new_set_v;
    } 
  }
  
  for (int i213 = 0; i213 <= 119; ++i213) {
    { float new_set_v = (n135_grad[i213 * 1 + 0] + mlp_grad[i213 * 1 + 0]);
      fprintf(log_file, "# n135.grad[i213, 0] := (n135.grad[i213, 0] + mlp.grad[i213, 0]);\n");
      fprintf(log_file, "n135_grad[%u] = %f = (n135_grad[%u]{=%f} + mlp_grad[%u]{=%f})\n", i213 * 1 + 0,
             new_set_v, i213 * 1 + 0, n135_grad[i213 * 1 + 0], i213 * 1 + 0, mlp_grad[i213 * 1 + 0]);
      fflush(log_file); n135_grad[i213 * 1 + 0] = new_set_v;
    } 
  }
  
  for (int i214 = 0; i214 <= 119; ++i214) {
    for (int i215 = 0; i215 <= 15; ++i215) {
      { float new_set_v = (w3_grad[0 * 16 + i215] + (n135_grad[i214 * 1 + 0] * n133[i214 * 16 + i215]));
        fprintf(log_file,
               "# w3.grad[0, i215] :=$  (w3.grad[0, i215] + (n135.grad[i214, 0] * n133[i214, i215]));\n");
        fprintf(log_file, "w3_grad[%u] = %f = (w3_grad[%u]{=%f} + (n135_grad[%u]{=%f} * n133[%u]{=%f}))\n",
               0 * 16 + i215, new_set_v, 0 * 16 + i215, w3_grad[0 * 16 + i215], i214 * 1 + 0,
               n135_grad[i214 * 1 + 0], i214 * 16 + i215, n133[i214 * 16 + i215]);
        fflush(log_file); w3_grad[0 * 16 + i215] = new_set_v;
      } 
    }
  }
  
  for (int i216 = 0; i216 <= 119; ++i216) {
    for (int i217 = 0; i217 <= 15; ++i217) {
      { float new_set_v = (n133_grad[i216 * 16 + i217] + (w3[0 * 16 + i217] * n135_grad[i216 * 1 + 0]));
        fprintf(log_file,
               "# n133.grad[i216, i217] :=$  (n133.grad[i216, i217] + (w3[0, i217] * n135.grad[i216, 0]));\n");
        fprintf(log_file, "n133_grad[%u] = %f = (n133_grad[%u]{=%f} + (w3[%u]{=%f} * n135_grad[%u]{=%f}))\n",
               i216 * 16 + i217, new_set_v, i216 * 16 + i217, n133_grad[i216 * 16 + i217], 0 * 16 + i217,
               w3[0 * 16 + i217], i216 * 1 + 0, n135_grad[i216 * 1 + 0]);
        fflush(log_file); n133_grad[i216 * 16 + i217] = new_set_v;
      } 
    }
  }
  
  for (int i218 = 0; i218 <= 119; ++i218) {
    for (int i219 = 0; i219 <= 15; ++i219) {
      {
        float new_set_v =
          (n131_grad[i218 * 16 + i219] + (n133[i218 * 16 + i219] > 0.0 ? n133_grad[i218 * 16 + i219] : 0.0));
        fprintf(log_file,
               "# n131.grad[i218, i219] :=$  (n131.grad[i218, i219] +$   (n133[i218, i219] > 0.0 ? n133.grad[i218, i219] : 0.0));\n");
        fprintf(log_file,
               "n131_grad[%u] = %f = (n131_grad[%u]{=%f} + (n133[%u]{=%f} > 0.0 ? n133_grad[%u]{=%f} : 0.0))\n",
               i218 * 16 + i219, new_set_v, i218 * 16 + i219, n131_grad[i218 * 16 + i219], i218 * 16 + i219,
               n133[i218 * 16 + i219], i218 * 16 + i219, n133_grad[i218 * 16 + i219]);
        fflush(log_file); n131_grad[i218 * 16 + i219] = new_set_v;
      } 
    }
  }
  
  for (int i220 = 0; i220 <= 119; ++i220) {
    for (int i221 = 0; i221 <= 15; ++i221) {
      { float new_set_v = (b2_grad[i221] + n131_grad[i220 * 16 + i221]);
        fprintf(log_file, "# b2.grad[i221] := (b2.grad[i221] + n131.grad[i220, i221]);\n");
        fprintf(log_file, "b2_grad[%u] = %f = (b2_grad[%u]{=%f} + n131_grad[%u]{=%f})\n", i221, new_set_v,
               i221, b2_grad[i221], i220 * 16 + i221, n131_grad[i220 * 16 + i221]);
        fflush(log_file); b2_grad[i221] = new_set_v;
      } 
    }
  }
  
  for (int i222 = 0; i222 <= 119; ++i222) {
    for (int i223 = 0; i223 <= 15; ++i223) {
      { float new_set_v = (n129_grad[i222 * 16 + i223] + n131_grad[i222 * 16 + i223]);
        fprintf(log_file, "# n129.grad[i222, i223] := (n129.grad[i222, i223] + n131.grad[i222, i223]);\n");
        fprintf(log_file, "n129_grad[%u] = %f = (n129_grad[%u]{=%f} + n131_grad[%u]{=%f})\n",
               i222 * 16 + i223, new_set_v, i222 * 16 + i223, n129_grad[i222 * 16 + i223], i222 * 16 + i223,
               n131_grad[i222 * 16 + i223]);
        fflush(log_file); n129_grad[i222 * 16 + i223] = new_set_v;
      } 
    }
  }
  
  for (int i224 = 0; i224 <= 119; ++i224) {
    for (int i225 = 0; i225 <= 15; ++i225) {
      for (int i226 = 0; i226 <= 15; ++i226) {
        {
          float new_set_v =
            (w2_grad[i225 * 16 + i226] + (n129_grad[i224 * 16 + i225] * n127[i224 * 16 + i226]));
          fprintf(log_file,
                 "# w2.grad[i225, i226] :=$  (w2.grad[i225, i226] + (n129.grad[i224, i225] * n127[i224, i226]));\n");
          fprintf(log_file, "w2_grad[%u] = %f = (w2_grad[%u]{=%f} + (n129_grad[%u]{=%f} * n127[%u]{=%f}))\n",
                 i225 * 16 + i226, new_set_v, i225 * 16 + i226, w2_grad[i225 * 16 + i226], i224 * 16 + i225,
                 n129_grad[i224 * 16 + i225], i224 * 16 + i226, n127[i224 * 16 + i226]);
          fflush(log_file); w2_grad[i225 * 16 + i226] = new_set_v;
        } 
      }
    }
  }
  
  for (int i227 = 0; i227 <= 119; ++i227) {
    for (int i228 = 0; i228 <= 15; ++i228) {
      for (int i229 = 0; i229 <= 15; ++i229) {
        {
          float new_set_v =
            (n127_grad[i227 * 16 + i229] + (w2[i228 * 16 + i229] * n129_grad[i227 * 16 + i228]));
          fprintf(log_file,
                 "# n127.grad[i227, i229] :=$  (n127.grad[i227, i229] + (w2[i228, i229] * n129.grad[i227, i228]));\n");
          fprintf(log_file,
                 "n127_grad[%u] = %f = (n127_grad[%u]{=%f} + (w2[%u]{=%f} * n129_grad[%u]{=%f}))\n",
                 i227 * 16 + i229, new_set_v, i227 * 16 + i229, n127_grad[i227 * 16 + i229], i228 * 16 + i229
                 , w2[i228 * 16 + i229], i227 * 16 + i228, n129_grad[i227 * 16 + i228]);
          fflush(log_file); n127_grad[i227 * 16 + i229] = new_set_v;
        } 
      }
    }
  }
  
  for (int i230 = 0; i230 <= 119; ++i230) {
    for (int i231 = 0; i231 <= 15; ++i231) {
      {
        float new_set_v =
          (n125_grad[i230 * 16 + i231] + (n127[i230 * 16 + i231] > 0.0 ? n127_grad[i230 * 16 + i231] : 0.0));
        fprintf(log_file,
               "# n125.grad[i230, i231] :=$  (n125.grad[i230, i231] +$   (n127[i230, i231] > 0.0 ? n127.grad[i230, i231] : 0.0));\n");
        fprintf(log_file,
               "n125_grad[%u] = %f = (n125_grad[%u]{=%f} + (n127[%u]{=%f} > 0.0 ? n127_grad[%u]{=%f} : 0.0))\n",
               i230 * 16 + i231, new_set_v, i230 * 16 + i231, n125_grad[i230 * 16 + i231], i230 * 16 + i231,
               n127[i230 * 16 + i231], i230 * 16 + i231, n127_grad[i230 * 16 + i231]);
        fflush(log_file); n125_grad[i230 * 16 + i231] = new_set_v;
      } 
    }
  }
  
  for (int i232 = 0; i232 <= 119; ++i232) {
    for (int i233 = 0; i233 <= 15; ++i233) {
      { float new_set_v = (b1_grad[i233] + n125_grad[i232 * 16 + i233]);
        fprintf(log_file, "# b1.grad[i233] := (b1.grad[i233] + n125.grad[i232, i233]);\n");
        fprintf(log_file, "b1_grad[%u] = %f = (b1_grad[%u]{=%f} + n125_grad[%u]{=%f})\n", i233, new_set_v,
               i233, b1_grad[i233], i232 * 16 + i233, n125_grad[i232 * 16 + i233]);
        fflush(log_file); b1_grad[i233] = new_set_v;
      } 
    }
  }
  
  for (int i234 = 0; i234 <= 119; ++i234) {
    for (int i235 = 0; i235 <= 15; ++i235) {
      { float new_set_v = (n123_grad[i234 * 16 + i235] + n125_grad[i234 * 16 + i235]);
        fprintf(log_file, "# n123.grad[i234, i235] := (n123.grad[i234, i235] + n125.grad[i234, i235]);\n");
        fprintf(log_file, "n123_grad[%u] = %f = (n123_grad[%u]{=%f} + n125_grad[%u]{=%f})\n",
               i234 * 16 + i235, new_set_v, i234 * 16 + i235, n123_grad[i234 * 16 + i235], i234 * 16 + i235,
               n125_grad[i234 * 16 + i235]);
        fflush(log_file); n123_grad[i234 * 16 + i235] = new_set_v;
      } 
    }
  }
  
  for (int i236 = 0; i236 <= 119; ++i236) {
    for (int i237 = 0; i237 <= 15; ++i237) {
      for (int i238 = 0; i238 <= 1; ++i238) {
        {
          float new_set_v =
            (w1_grad[i237 * 2 + i238] + (n123_grad[i236 * 16 + i237] * input[i236 * 2 + i238]));
          fprintf(log_file,
                 "# w1.grad[i237, i238] :=$  (w1.grad[i237, i238] + (n123.grad[i236, i237] * input[i236, i238]));\n");
          fprintf(log_file,
                 "w1_grad[%u] = %f = (w1_grad[%u]{=%f} + (n123_grad[%u]{=%f} * input[%u]{=%f}))\n",
                 i237 * 2 + i238, new_set_v, i237 * 2 + i238, w1_grad[i237 * 2 + i238], i236 * 16 + i237,
                 n123_grad[i236 * 16 + i237], i236 * 2 + i238, input[i236 * 2 + i238]);
          fflush(log_file); w1_grad[i237 * 2 + i238] = new_set_v;
        } 
      }
    }
  }
  
  fprintf(log_file,
  "COMMENT: end\n");
  
  fprintf(log_file,
  "COMMENT: end\n");
  
}

