extern void
infer_mlp (const char * log_file_name)
{
  FILE * log_file;
  float[16] n412;
  float[16] n414;
  float[16] n416;
  float[16] n418;
  float[16] n420;
  float[16] n422;
  float[1] n424;
  int i609;
  int i610;
  int i612;
  int i614;
  int i617;
  int i618;
  int i620;
  int i622;
  int i624;

init_infer_mlp:
  log_file = fopen (log_file_name, "w");
  /* Array #412 *: Local_only; ptr: "(float *)&n412". */
  (void)fprintf (log_file, "memset_zero(n412) where before first element = %g\n", ((double)((float *)&n412)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n412), (int)0, (size_t)64);
  /* Array #327 w1: From_context; ptr: "(float *)0x5637b75e4600". */
  /* Array #410 infer: From_context; ptr: "(float *)0x5637b7536fe0". */
  /* Array #414 +: Local_only; ptr: "(float *)&n414". */
  /* Array #321 b1: From_context; ptr: "(float *)0x5637b7715f50". */
  /* Array #416 ?/: Local_only; ptr: "(float *)&n416". */
  /* Array #418 *: Local_only; ptr: "(float *)&n418". */
  (void)fprintf (log_file, "memset_zero(n418) where before first element = %g\n", ((double)((float *)&n418)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n418), (int)0, (size_t)64);
  /* Array #329 w2: From_context; ptr: "(float *)0x5637b7541570". */
  /* Array #420 +: Local_only; ptr: "(float *)&n420". */
  /* Array #323 b2: From_context; ptr: "(float *)0x5637b7e565d0". */
  /* Array #422 ?/: Local_only; ptr: "(float *)&n422". */
  /* Array #424 *: Local_only; ptr: "(float *)&n424". */
  (void)fprintf (log_file, "memset_zero(n424) where before first element = %g\n", ((double)((float *)&n424)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n424), (int)0, (size_t)4);
  /* Array #331 w3: From_context; ptr: "(float *)0x5637b7859f50". */
  /* Array #426 +_mlp: From_context; ptr: "(float *)0x5637b7eaa760". */
  /* Array #325 b3: From_context; ptr: "(float *)0x5637b7711e00". */
  goto infer_mlp;

infer_mlp:
  (void)fprintf (log_file, "\nCOMMENT: infer mlp\n");
  (void)fflush (log_file);
  (void)fprintf (log_file, "memset_zero(n412) where before first element = %g\n", ((double)((float *)&n412)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n412), (int)0, (size_t)64);
  i609 = (int)0;
  goto loop_cond_i609;

loop_cond_i609:
  if (i609 > (int)15) goto after_loop_i609; else goto loop_body_i609;

loop_body_i609:
  (void)fprintf (log_file, "index i609 = %d\n", i609);
  (void)fflush (log_file);
  i610 = (int)0;
  goto loop_cond_i610;

after_loop_i609:
  i612 = (int)0;
  goto loop_cond_i612;

loop_cond_i610:
  if (i610 > (int)1) goto after_loop_i610; else goto loop_body_i610;

loop_body_i610:
  (void)fprintf (log_file, "index i610 = %d\n", i610);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n412[i609] := (n412[i609] + (w1[i609, i610] * infer[i610]));\n");
  (void)fprintf (log_file, "n412[%d]{=%g} += %g = (w1[%d]{=%g} * infer[%d]{=%g})\n", ((double)((float *)&n412)[(i609 + (int)0 * (int)16)]), (i609 + (int)0 * (int)16), ((double)((float *)0x5637b75e4600[(i610 + (i609 + (int)0 * (int)16) * (int)2)] * (float *)0x5637b7536fe0[(i610 + (int)0 * (int)2)])), (i610 + (i609 + (int)0 * (int)16) * (int)2), ((double)(float *)0x5637b75e4600[(i610 + (i609 + (int)0 * (int)16) * (int)2)]), (i610 + (int)0 * (int)2), ((double)(float *)0x5637b7536fe0[(i610 + (int)0 * (int)2)]));
  (void)fflush (log_file);
  ((float *)&n412)[(i609 + (int)0 * (int)16)] += (float *)0x5637b75e4600[(i610 + (i609 + (int)0 * (int)16) * (int)2)] * (float *)0x5637b7536fe0[(i610 + (int)0 * (int)2)];
  i610 += (int)1;
  goto loop_cond_i610;

after_loop_i610:
  i609 += (int)1;
  goto loop_cond_i609;

loop_cond_i612:
  if (i612 > (int)15) goto after_loop_i612; else goto loop_body_i612;

loop_body_i612:
  (void)fprintf (log_file, "index i612 = %d\n", i612);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n414[i612] := (b1[i612] + n412[i612]);\n");
  (void)fprintf (log_file, "n414[%d]{=%g} = %g = (b1[%d]{=%g} + n412[%d]{=%g})\n", ((double)((float *)&n414)[(i612 + (int)0 * (int)16)]), (i612 + (int)0 * (int)16), ((double)((float *)0x5637b7715f50[(i612 + (int)0 * (int)16)] + ((float *)&n412)[(i612 + (int)0 * (int)16)])), (i612 + (int)0 * (int)16), ((double)(float *)0x5637b7715f50[(i612 + (int)0 * (int)16)]), (i612 + (int)0 * (int)16), ((double)((float *)&n412)[(i612 + (int)0 * (int)16)]));
  (void)fflush (log_file);
  ((float *)&n414)[(i612 + (int)0 * (int)16)] = (float *)0x5637b7715f50[(i612 + (int)0 * (int)16)] + ((float *)&n412)[(i612 + (int)0 * (int)16)];
  i612 += (int)1;
  goto loop_cond_i612;

after_loop_i612:
  i614 = (int)0;
  goto loop_cond_i614;

loop_cond_i614:
  if (i614 > (int)15) goto after_loop_i614; else goto loop_body_i614;

loop_body_i614:
  (void)fprintf (log_file, "index i614 = %d\n", i614);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n416[i614] := relu(n414[i614]);\n");
  (void)fprintf (log_file, "n416[%d]{=%g} = %g = (n414[%d]{=%g} > 0.0 ? n414[%d]{=%g} : 0.0)\n", ((double)((float *)&n416)[(i614 + (int)0 * (int)16)]), (i614 + (int)0 * (int)16), ((double)((float)(int)((float)0 < ((float *)&n414)[(i614 + (int)0 * (int)16)]) * ((float *)&n414)[(i614 + (int)0 * (int)16)])), (i614 + (int)0 * (int)16), ((double)((float *)&n414)[(i614 + (int)0 * (int)16)]), (i614 + (int)0 * (int)16), ((double)((float *)&n414)[(i614 + (int)0 * (int)16)]));
  (void)fflush (log_file);
  ((float *)&n416)[(i614 + (int)0 * (int)16)] = (float)(int)((float)0 < ((float *)&n414)[(i614 + (int)0 * (int)16)]) * ((float *)&n414)[(i614 + (int)0 * (int)16)];
  i614 += (int)1;
  goto loop_cond_i614;

after_loop_i614:
  (void)fprintf (log_file, "memset_zero(n418) where before first element = %g\n", ((double)((float *)&n418)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n418), (int)0, (size_t)64);
  i617 = (int)0;
  goto loop_cond_i617;

loop_cond_i617:
  if (i617 > (int)15) goto after_loop_i617; else goto loop_body_i617;

loop_body_i617:
  (void)fprintf (log_file, "index i617 = %d\n", i617);
  (void)fflush (log_file);
  i618 = (int)0;
  goto loop_cond_i618;

after_loop_i617:
  i620 = (int)0;
  goto loop_cond_i620;

loop_cond_i618:
  if (i618 > (int)15) goto after_loop_i618; else goto loop_body_i618;

loop_body_i618:
  (void)fprintf (log_file, "index i618 = %d\n", i618);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n418[i617] := (n418[i617] + (w2[i617, i618] * n416[i618]));\n");
  (void)fprintf (log_file, "n418[%d]{=%g} += %g = (w2[%d]{=%g} * n416[%d]{=%g})\n", ((double)((float *)&n418)[(i617 + (int)0 * (int)16)]), (i617 + (int)0 * (int)16), ((double)((float *)0x5637b7541570[(i618 + (i617 + (int)0 * (int)16) * (int)16)] * ((float *)&n416)[(i618 + (int)0 * (int)16)])), (i618 + (i617 + (int)0 * (int)16) * (int)16), ((double)(float *)0x5637b7541570[(i618 + (i617 + (int)0 * (int)16) * (int)16)]), (i618 + (int)0 * (int)16), ((double)((float *)&n416)[(i618 + (int)0 * (int)16)]));
  (void)fflush (log_file);
  ((float *)&n418)[(i617 + (int)0 * (int)16)] += (float *)0x5637b7541570[(i618 + (i617 + (int)0 * (int)16) * (int)16)] * ((float *)&n416)[(i618 + (int)0 * (int)16)];
  i618 += (int)1;
  goto loop_cond_i618;

after_loop_i618:
  i617 += (int)1;
  goto loop_cond_i617;

loop_cond_i620:
  if (i620 > (int)15) goto after_loop_i620; else goto loop_body_i620;

loop_body_i620:
  (void)fprintf (log_file, "index i620 = %d\n", i620);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n420[i620] := (b2[i620] + n418[i620]);\n");
  (void)fprintf (log_file, "n420[%d]{=%g} = %g = (b2[%d]{=%g} + n418[%d]{=%g})\n", ((double)((float *)&n420)[(i620 + (int)0 * (int)16)]), (i620 + (int)0 * (int)16), ((double)((float *)0x5637b7e565d0[(i620 + (int)0 * (int)16)] + ((float *)&n418)[(i620 + (int)0 * (int)16)])), (i620 + (int)0 * (int)16), ((double)(float *)0x5637b7e565d0[(i620 + (int)0 * (int)16)]), (i620 + (int)0 * (int)16), ((double)((float *)&n418)[(i620 + (int)0 * (int)16)]));
  (void)fflush (log_file);
  ((float *)&n420)[(i620 + (int)0 * (int)16)] = (float *)0x5637b7e565d0[(i620 + (int)0 * (int)16)] + ((float *)&n418)[(i620 + (int)0 * (int)16)];
  i620 += (int)1;
  goto loop_cond_i620;

after_loop_i620:
  i622 = (int)0;
  goto loop_cond_i622;

loop_cond_i622:
  if (i622 > (int)15) goto after_loop_i622; else goto loop_body_i622;

loop_body_i622:
  (void)fprintf (log_file, "index i622 = %d\n", i622);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n422[i622] := relu(n420[i622]);\n");
  (void)fprintf (log_file, "n422[%d]{=%g} = %g = (n420[%d]{=%g} > 0.0 ? n420[%d]{=%g} : 0.0)\n", ((double)((float *)&n422)[(i622 + (int)0 * (int)16)]), (i622 + (int)0 * (int)16), ((double)((float)(int)((float)0 < ((float *)&n420)[(i622 + (int)0 * (int)16)]) * ((float *)&n420)[(i622 + (int)0 * (int)16)])), (i622 + (int)0 * (int)16), ((double)((float *)&n420)[(i622 + (int)0 * (int)16)]), (i622 + (int)0 * (int)16), ((double)((float *)&n420)[(i622 + (int)0 * (int)16)]));
  (void)fflush (log_file);
  ((float *)&n422)[(i622 + (int)0 * (int)16)] = (float)(int)((float)0 < ((float *)&n420)[(i622 + (int)0 * (int)16)]) * ((float *)&n420)[(i622 + (int)0 * (int)16)];
  i622 += (int)1;
  goto loop_cond_i622;

after_loop_i622:
  (void)fprintf (log_file, "memset_zero(n424) where before first element = %g\n", ((double)((float *)&n424)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n424), (int)0, (size_t)4);
  i624 = (int)0;
  goto loop_cond_i624;

loop_cond_i624:
  if (i624 > (int)15) goto after_loop_i624; else goto loop_body_i624;

loop_body_i624:
  (void)fprintf (log_file, "index i624 = %d\n", i624);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n424[0] := (n424[0] + (w3[0, i624] * n422[i624]));\n");
  (void)fprintf (log_file, "n424[%d]{=%g} += %g = (w3[%d]{=%g} * n422[%d]{=%g})\n", ((double)((float *)&n424)[((int)0 + (int)0 * (int)1)]), ((int)0 + (int)0 * (int)1), ((double)((float *)0x5637b7859f50[(i624 + ((int)0 + (int)0 * (int)1) * (int)16)] * ((float *)&n422)[(i624 + (int)0 * (int)16)])), (i624 + ((int)0 + (int)0 * (int)1) * (int)16), ((double)(float *)0x5637b7859f50[(i624 + ((int)0 + (int)0 * (int)1) * (int)16)]), (i624 + (int)0 * (int)16), ((double)((float *)&n422)[(i624 + (int)0 * (int)16)]));
  (void)fflush (log_file);
  ((float *)&n424)[((int)0 + (int)0 * (int)1)] += (float *)0x5637b7859f50[(i624 + ((int)0 + (int)0 * (int)1) * (int)16)] * ((float *)&n422)[(i624 + (int)0 * (int)16)];
  i624 += (int)1;
  goto loop_cond_i624;

after_loop_i624:
  (void)fprintf (log_file, "# mlp[0] := (b3[0] + n424[0]);\n");
  (void)fprintf (log_file, "mlp[%d]{=%g} = %g = (b3[%d]{=%g} + n424[%d]{=%g})\n", ((double)(float *)0x5637b7eaa760[((int)0 + (int)0 * (int)1)]), ((int)0 + (int)0 * (int)1), ((double)((float *)0x5637b7711e00[((int)0 + (int)0 * (int)1)] + ((float *)&n424)[((int)0 + (int)0 * (int)1)])), ((int)0 + (int)0 * (int)1), ((double)(float *)0x5637b7711e00[((int)0 + (int)0 * (int)1)]), ((int)0 + (int)0 * (int)1), ((double)((float *)&n424)[((int)0 + (int)0 * (int)1)]));
  (void)fflush (log_file);
  (float *)0x5637b7eaa760[((int)0 + (int)0 * (int)1)] = (float *)0x5637b7711e00[((int)0 + (int)0 * (int)1)] + ((float *)&n424)[((int)0 + (int)0 * (int)1)];
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

