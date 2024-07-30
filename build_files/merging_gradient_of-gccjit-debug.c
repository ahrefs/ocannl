extern void
merging_gradient_of_b1 (float * b1_grad, const char * log_file_name, float * merge_buffer)
{
  FILE * log_file;
  int i764;

init_merging_gradient_of_b1:
  log_file = fopen (log_file_name, "w");
  /* Array #429 grad_b1: From_context; ptr: b1_grad. */
  goto merging_gradient_of_b1;

merging_gradient_of_b1:
  (void)fprintf (log_file, "\nCOMMENT: merging gradient of b1\n");
  (void)fflush (log_file);
  i764 = (int)0;
  goto loop_cond_i764;

loop_cond_i764:
  if (i764 > (int)15) goto after_loop_i764; else goto loop_body_i764;

loop_body_i764:
  (void)fprintf (log_file, "index i764 = %d\n", i764);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# b1.grad[i764] := (b1.grad[i764] + b1.grad.merge[i764]);\n");
  (void)fprintf (log_file, "b1_grad[%d]{=%g} += %g = b1_grad.merge[%d]{=%g}\n", ((double)b1_grad[(i764 + (int)0 * (int)16)]), (i764 + (int)0 * (int)16), ((double)merge_buffer[(i764 + (int)0 * (int)16)]), (i764 + (int)0 * (int)16), ((double)merge_buffer[(i764 + (int)0 * (int)16)]));
  (void)fflush (log_file);
  b1_grad[(i764 + (int)0 * (int)16)] += merge_buffer[(i764 + (int)0 * (int)16)];
  i764 += (int)1;
  goto loop_cond_i764;

after_loop_i764:
  (void)fprintf (log_file, "\nCOMMENT: end\n");
  (void)fflush (log_file);
  (void)fclose (log_file);
  return;
}

extern FILE *
fopen (const char * filename, const char * mode); /* (imported) */

extern void
fflush (FILE * f); /* (imported) */

extern void *
fclose (FILE * f); /* (imported) */

extern void
merging_gradient_of_b2 (float * b2_grad, const char * log_file_name, float * merge_buffer)
{
  FILE * log_file;
  int i766;

init_merging_gradient_of_b2:
  log_file = fopen (log_file_name, "w");
  /* Array #431 grad_b2: From_context; ptr: b2_grad. */
  goto merging_gradient_of_b2;

merging_gradient_of_b2:
  (void)fprintf (log_file, "\nCOMMENT: merging gradient of b2\n");
  (void)fflush (log_file);
  i766 = (int)0;
  goto loop_cond_i766;

loop_cond_i766:
  if (i766 > (int)15) goto after_loop_i766; else goto loop_body_i766;

loop_body_i766:
  (void)fprintf (log_file, "index i766 = %d\n", i766);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# b2.grad[i766] := (b2.grad[i766] + b2.grad.merge[i766]);\n");
  (void)fprintf (log_file, "b2_grad[%d]{=%g} += %g = b2_grad.merge[%d]{=%g}\n", ((double)b2_grad[(i766 + (int)0 * (int)16)]), (i766 + (int)0 * (int)16), ((double)merge_buffer[(i766 + (int)0 * (int)16)]), (i766 + (int)0 * (int)16), ((double)merge_buffer[(i766 + (int)0 * (int)16)]));
  (void)fflush (log_file);
  b2_grad[(i766 + (int)0 * (int)16)] += merge_buffer[(i766 + (int)0 * (int)16)];
  i766 += (int)1;
  goto loop_cond_i766;

after_loop_i766:
  (void)fprintf (log_file, "\nCOMMENT: end\n");
  (void)fflush (log_file);
  (void)fclose (log_file);
  return;
}

extern FILE *
fopen (const char * filename, const char * mode); /* (imported) */

extern void
fflush (FILE * f); /* (imported) */

extern void *
fclose (FILE * f); /* (imported) */

extern void
merging_gradient_of_b3 (float * b3_grad, const char * log_file_name, float * merge_buffer)
{
  FILE * log_file;

init_merging_gradient_of_b3:
  log_file = fopen (log_file_name, "w");
  /* Array #433 grad_b3: From_context; ptr: b3_grad. */
  goto merging_gradient_of_b3;

merging_gradient_of_b3:
  (void)fprintf (log_file, "\nCOMMENT: merging gradient of b3\n");
  (void)fflush (log_file);
  (void)fprintf (log_file, "# b3.grad[0] := (b3.grad[0] + b3.grad.merge[0]);\n");
  (void)fprintf (log_file, "b3_grad[%d]{=%g} += %g = b3_grad.merge[%d]{=%g}\n", ((double)b3_grad[((int)0 + (int)0 * (int)1)]), ((int)0 + (int)0 * (int)1), ((double)merge_buffer[((int)0 + (int)0 * (int)1)]), ((int)0 + (int)0 * (int)1), ((double)merge_buffer[((int)0 + (int)0 * (int)1)]));
  (void)fflush (log_file);
  b3_grad[((int)0 + (int)0 * (int)1)] += merge_buffer[((int)0 + (int)0 * (int)1)];
  (void)fprintf (log_file, "\nCOMMENT: end\n");
  (void)fflush (log_file);
  (void)fclose (log_file);
  return;
}

extern FILE *
fopen (const char * filename, const char * mode); /* (imported) */

extern void
fflush (FILE * f); /* (imported) */

extern void *
fclose (FILE * f); /* (imported) */

extern void
merging_gradient_of_w1 (float * w1_grad, const char * log_file_name, float * merge_buffer)
{
  FILE * log_file;
  int i769;
  int i770;

init_merging_gradient_of_w1:
  log_file = fopen (log_file_name, "w");
  /* Array #435 grad_w1: From_context; ptr: w1_grad. */
  goto merging_gradient_of_w1;

merging_gradient_of_w1:
  (void)fprintf (log_file, "\nCOMMENT: merging gradient of w1\n");
  (void)fflush (log_file);
  i769 = (int)0;
  goto loop_cond_i769;

loop_cond_i769:
  if (i769 > (int)15) goto after_loop_i769; else goto loop_body_i769;

loop_body_i769:
  (void)fprintf (log_file, "index i769 = %d\n", i769);
  (void)fflush (log_file);
  i770 = (int)0;
  goto loop_cond_i770;

after_loop_i769:
  (void)fprintf (log_file, "\nCOMMENT: end\n");
  (void)fflush (log_file);
  (void)fclose (log_file);
  return;

loop_cond_i770:
  if (i770 > (int)1) goto after_loop_i770; else goto loop_body_i770;

loop_body_i770:
  (void)fprintf (log_file, "index i770 = %d\n", i770);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# w1.grad[i769, i770] := (w1.grad[i769, i770] + w1.grad.merge[i769, i770]);\n");
  (void)fprintf (log_file, "w1_grad[%d]{=%g} += %g = w1_grad.merge[%d]{=%g}\n", ((double)w1_grad[(i770 + (i769 + (int)0 * (int)16) * (int)2)]), (i770 + (i769 + (int)0 * (int)16) * (int)2), ((double)merge_buffer[(i770 + (i769 + (int)0 * (int)16) * (int)2)]), (i770 + (i769 + (int)0 * (int)16) * (int)2), ((double)merge_buffer[(i770 + (i769 + (int)0 * (int)16) * (int)2)]));
  (void)fflush (log_file);
  w1_grad[(i770 + (i769 + (int)0 * (int)16) * (int)2)] += merge_buffer[(i770 + (i769 + (int)0 * (int)16) * (int)2)];
  i770 += (int)1;
  goto loop_cond_i770;

after_loop_i770:
  i769 += (int)1;
  goto loop_cond_i769;
}

extern FILE *
fopen (const char * filename, const char * mode); /* (imported) */

extern void
fflush (FILE * f); /* (imported) */

extern void *
fclose (FILE * f); /* (imported) */

extern void
merging_gradient_of_w2 (float * w2_grad, const char * log_file_name, float * merge_buffer)
{
  FILE * log_file;
  int i773;
  int i774;

init_merging_gradient_of_w2:
  log_file = fopen (log_file_name, "w");
  /* Array #437 grad_w2: From_context; ptr: w2_grad. */
  goto merging_gradient_of_w2;

merging_gradient_of_w2:
  (void)fprintf (log_file, "\nCOMMENT: merging gradient of w2\n");
  (void)fflush (log_file);
  i773 = (int)0;
  goto loop_cond_i773;

loop_cond_i773:
  if (i773 > (int)15) goto after_loop_i773; else goto loop_body_i773;

loop_body_i773:
  (void)fprintf (log_file, "index i773 = %d\n", i773);
  (void)fflush (log_file);
  i774 = (int)0;
  goto loop_cond_i774;

after_loop_i773:
  (void)fprintf (log_file, "\nCOMMENT: end\n");
  (void)fflush (log_file);
  (void)fclose (log_file);
  return;

loop_cond_i774:
  if (i774 > (int)15) goto after_loop_i774; else goto loop_body_i774;

loop_body_i774:
  (void)fprintf (log_file, "index i774 = %d\n", i774);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# w2.grad[i773, i774] := (w2.grad[i773, i774] + w2.grad.merge[i773, i774]);\n");
  (void)fprintf (log_file, "w2_grad[%d]{=%g} += %g = w2_grad.merge[%d]{=%g}\n", ((double)w2_grad[(i774 + (i773 + (int)0 * (int)16) * (int)16)]), (i774 + (i773 + (int)0 * (int)16) * (int)16), ((double)merge_buffer[(i774 + (i773 + (int)0 * (int)16) * (int)16)]), (i774 + (i773 + (int)0 * (int)16) * (int)16), ((double)merge_buffer[(i774 + (i773 + (int)0 * (int)16) * (int)16)]));
  (void)fflush (log_file);
  w2_grad[(i774 + (i773 + (int)0 * (int)16) * (int)16)] += merge_buffer[(i774 + (i773 + (int)0 * (int)16) * (int)16)];
  i774 += (int)1;
  goto loop_cond_i774;

after_loop_i774:
  i773 += (int)1;
  goto loop_cond_i773;
}

extern FILE *
fopen (const char * filename, const char * mode); /* (imported) */

extern void
fflush (FILE * f); /* (imported) */

extern void *
fclose (FILE * f); /* (imported) */

extern void
merging_gradient_of_w3 (float * w3_grad, const char * log_file_name, float * merge_buffer)
{
  FILE * log_file;
  int i776;

init_merging_gradient_of_w3:
  log_file = fopen (log_file_name, "w");
  /* Array #439 grad_w3: From_context; ptr: w3_grad. */
  goto merging_gradient_of_w3;

merging_gradient_of_w3:
  (void)fprintf (log_file, "\nCOMMENT: merging gradient of w3\n");
  (void)fflush (log_file);
  i776 = (int)0;
  goto loop_cond_i776;

loop_cond_i776:
  if (i776 > (int)15) goto after_loop_i776; else goto loop_body_i776;

loop_body_i776:
  (void)fprintf (log_file, "index i776 = %d\n", i776);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# w3.grad[0, i776] := (w3.grad[0, i776] + w3.grad.merge[0, i776]);\n");
  (void)fprintf (log_file, "w3_grad[%d]{=%g} += %g = w3_grad.merge[%d]{=%g}\n", ((double)w3_grad[(i776 + ((int)0 + (int)0 * (int)1) * (int)16)]), (i776 + ((int)0 + (int)0 * (int)1) * (int)16), ((double)merge_buffer[(i776 + ((int)0 + (int)0 * (int)1) * (int)16)]), (i776 + ((int)0 + (int)0 * (int)1) * (int)16), ((double)merge_buffer[(i776 + ((int)0 + (int)0 * (int)1) * (int)16)]));
  (void)fflush (log_file);
  w3_grad[(i776 + ((int)0 + (int)0 * (int)1) * (int)16)] += merge_buffer[(i776 + ((int)0 + (int)0 * (int)1) * (int)16)];
  i776 += (int)1;
  goto loop_cond_i776;

after_loop_i776:
  (void)fprintf (log_file, "\nCOMMENT: end\n");
  (void)fflush (log_file);
  (void)fclose (log_file);
  return;
}

extern FILE *
fopen (const char * filename, const char * mode); /* (imported) */

extern void
fflush (FILE * f); /* (imported) */

extern void *
fclose (FILE * f); /* (imported) */

