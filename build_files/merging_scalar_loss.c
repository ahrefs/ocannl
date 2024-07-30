#include <stdio.h>
#include <stdlib.h>
/* Global declarations. */
#define scalar_loss ((float*)0x5593ecda63a0)

void merging_scalar_loss(const char* log_file_name, const float *merge_buffer) {
  FILE* log_file = fopen(log_file_name, "w");
  /* Debug initial parameter state. */
  fprintf(log_file, "const float *merge_buffer = %p\n", (void*)merge_buffer);
  /* Local declarations and initialization. */
  
  /* Main logic. */
  fprintf(log_file,
  "COMMENT: merging scalar_loss\n");
  
  { float new_set_v = (scalar_loss[0] + ((float*)merge_buffer)[0]);
    fprintf(log_file, "# scalar_loss[0] := (scalar_loss[0] + scalar_loss.merge[0]);\n");
    fprintf(log_file, "scalar_loss[%u] = %f = (scalar_loss[%u]{=%f} + merge_buffer[%u]{=%f})\n", 0,
           new_set_v, 0, scalar_loss[0], 0, merge_buffer[0]);
    fflush(log_file); scalar_loss[0] = new_set_v;
  }
  
  fprintf(log_file,
  "COMMENT: end\n");
  
}

