#include <stdio.h>
#include <stdlib.h>
/* Global declarations. */
#define b3 ((float*)0x5593ecda63a0)
#define w1 ((float*)0x5593ecdaafc0)
#define b2 ((float*)0x5593ecda99c0)
#define mlp ((float*)0x5593ece52cb0)
#define w3 ((float*)0x5593ecda9a60)
#define b1 ((float*)0x5593ecdad370)
#define infer ((float*)0x5593ecdd9e50)
#define w2 ((float*)0x5593ecdab0e0)

void infer_mlp(const char* log_file_name) {
  FILE* log_file = fopen(log_file_name, "w");
  /* Debug initial parameter state. */
  /* Local declarations and initialization. */
  float n93[16];
  float n95[16];
  float n97[16] = {0};
  float n99[16];
  float n91[16] = {0};
  float n103[1] = {0};
  float n101[16];
  
  /* Main logic. */
  fprintf(log_file,
  "COMMENT: infer mlp\n");
  
  
  for (int i141 = 0; i141 <= 15; ++i141) {
    for (int i142 = 0; i142 <= 1; ++i142) {
      { float new_set_v = (n91[i141] + (w1[i141 * 2 + i142] * infer[i142]));
        fprintf(log_file, "# n91[i141] := (n91[i141] + (w1[i141, i142] * infer[i142]));\n");
        fprintf(log_file, "n91[%u] = %f = (n91[%u]{=%f} + (w1[%u]{=%f} * infer[%u]{=%f}))\n", i141,
               new_set_v, i141, n91[i141], i141 * 2 + i142, w1[i141 * 2 + i142], i142, infer[i142]);
        fflush(log_file); n91[i141] = new_set_v;
      } 
    }
  }
  
  for (int i144 = 0; i144 <= 15; ++i144) {
    { float new_set_v = (b1[i144] + n91[i144]);
      fprintf(log_file, "# n93[i144] := (b1[i144] + n91[i144]);\n");
      fprintf(log_file, "n93[%u] = %f = (b1[%u]{=%f} + n91[%u]{=%f})\n", i144, new_set_v, i144, b1[i144],
             i144, n91[i144]);
      fflush(log_file); n93[i144] = new_set_v;
    } 
  }
  
  for (int i146 = 0; i146 <= 15; ++i146) {
    { float new_set_v = (n93[i146] > 0.0 ? n93[i146] : 0.0);
      fprintf(log_file, "# n95[i146] := relu(n93[i146]);\n");
      fprintf(log_file, "n95[%u] = %f = (n93[%u]{=%f} > 0.0 ? n93[%u]{=%f} : 0.0)\n", i146, new_set_v, i146,
             n93[i146], i146, n93[i146]);
      fflush(log_file); n95[i146] = new_set_v;
    } 
  }
  
  
  for (int i149 = 0; i149 <= 15; ++i149) {
    for (int i150 = 0; i150 <= 15; ++i150) {
      { float new_set_v = (n97[i149] + (w2[i149 * 16 + i150] * n95[i150]));
        fprintf(log_file, "# n97[i149] := (n97[i149] + (w2[i149, i150] * n95[i150]));\n");
        fprintf(log_file, "n97[%u] = %f = (n97[%u]{=%f} + (w2[%u]{=%f} * n95[%u]{=%f}))\n", i149, new_set_v,
               i149, n97[i149], i149 * 16 + i150, w2[i149 * 16 + i150], i150, n95[i150]);
        fflush(log_file); n97[i149] = new_set_v;
      } 
    }
  }
  
  for (int i152 = 0; i152 <= 15; ++i152) {
    { float new_set_v = (b2[i152] + n97[i152]);
      fprintf(log_file, "# n99[i152] := (b2[i152] + n97[i152]);\n");
      fprintf(log_file, "n99[%u] = %f = (b2[%u]{=%f} + n97[%u]{=%f})\n", i152, new_set_v, i152, b2[i152],
             i152, n97[i152]);
      fflush(log_file); n99[i152] = new_set_v;
    } 
  }
  
  for (int i154 = 0; i154 <= 15; ++i154) {
    { float new_set_v = (n99[i154] > 0.0 ? n99[i154] : 0.0);
      fprintf(log_file, "# n101[i154] := relu(n99[i154]);\n");
      fprintf(log_file, "n101[%u] = %f = (n99[%u]{=%f} > 0.0 ? n99[%u]{=%f} : 0.0)\n", i154, new_set_v, i154,
             n99[i154], i154, n99[i154]);
      fflush(log_file); n101[i154] = new_set_v;
    } 
  }
  
  
  for (int i156 = 0; i156 <= 15; ++i156) {
    { float new_set_v = (n103[0] + (w3[0 * 16 + i156] * n101[i156]));
      fprintf(log_file, "# n103[0] := (n103[0] + (w3[0, i156] * n101[i156]));\n");
      fprintf(log_file, "n103[%u] = %f = (n103[%u]{=%f} + (w3[%u]{=%f} * n101[%u]{=%f}))\n", 0, new_set_v, 0,
             n103[0], 0 * 16 + i156, w3[0 * 16 + i156], i156, n101[i156]);
      fflush(log_file); n103[0] = new_set_v;
    } 
  }
  
  { float new_set_v = (b3[0] + n103[0]); fprintf(log_file, "# mlp[0] := (b3[0] + n103[0]);\n");
    fprintf(log_file, "mlp[%u] = %f = (b3[%u]{=%f} + n103[%u]{=%f})\n", 0, new_set_v, 0, b3[0], 0, n103[0]);
    fflush(log_file); mlp[0] = new_set_v;
  }
  
  fprintf(log_file,
  "COMMENT: end\n");
  
}

