extern void
merging_scalar_loss (const char * log_file_name, float * merge_buffer)
{
  FILE * log_file;

init_merging_scalar_loss:
  log_file = fopen (log_file_name, "w");
  /* Array #470 /._scalar_loss: From_context; ptr: "(float *)0x5637b76de010". */
  goto merging_scalar_loss;

merging_scalar_loss:
  (void)fprintf (log_file, "\nCOMMENT: merging scalar_loss\n");
  (void)fflush (log_file);
  (void)fprintf (log_file, "# scalar_loss[0] := (scalar_loss[0] + scalar_loss.merge[0]);\n");
  (void)fprintf (log_file, "scalar_loss[%d]{=%g} += %g = scalar_loss.merge[%d]{=%g}\n", ((double)(float *)0x5637b76de010[((int)0 + (int)0 * (int)1)]), ((int)0 + (int)0 * (int)1), ((double)merge_buffer[((int)0 + (int)0 * (int)1)]), ((int)0 + (int)0 * (int)1), ((double)merge_buffer[((int)0 + (int)0 * (int)1)]));
  (void)fflush (log_file);
  (float *)0x5637b76de010[((int)0 + (int)0 * (int)1)] += merge_buffer[((int)0 + (int)0 * (int)1)];
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

