extern void
scalar_loss_sgd_update (const char * log_file_name, int i625, int i626)
{
  FILE * log_file;
  float[16] n491;
  float[16] sgd_delta_b1;
  float[1] n479;
  float[1] n483;
  float[1] n485;
  float[16] n489;
  float[16] n496;
  float[16] sgd_delta_b2;
  float[16] n494;
  float[1] n501;
  float[1] sgd_delta_b3;
  float[1] n499;
  float[32] n506;
  float[32] sgd_delta_w1;
  float[32] n504;
  float[256] n511;
  float[256] sgd_delta_w2;
  float[256] n509;
  float[16] n516;
  float[16] sgd_delta_w3;
  float[16] n514;
  int i708;
  int i710;
  int i712;
  int i714;
  int i716;
  int i718;
  int i720;
  int i722;
  int i725;
  int i726;
  int i729;
  int i730;
  int i733;
  int i734;
  int i737;
  int i738;
  int i741;
  int i742;
  int i745;
  int i746;
  int i749;
  int i750;
  int i753;
  int i754;
  int i756;
  int i758;
  int i760;
  int i762;

init_scalar_loss_sgd_update:
  log_file = fopen (log_file_name, "w");
  (void)fprintf (log_file, "index i625 = %d\n", i625);
  (void)fflush (log_file);
  (void)fprintf (log_file, "index i626 = %d\n", i626);
  (void)fflush (log_file);
  /* Array #491 *.: Local_only; ptr: "(float *)&n491". */
  /* Array #428 b1: From_context; ptr: "(float *)0x5637b7e8f840". */
  /* Array #487 sgd_delta_b1: Local_only; ptr: "(float *)&sgd_delta_b1". */
  /* Array #429 grad_b1: From_context; ptr: "(float *)0x5637b79595d0". */
  /* Array #479 !@: Local_only; ptr: "(float *)&n479". */
  /* Array #483 -: Local_only; ptr: "(float *)&n483". */
  /* Array #485 *.: Local_only; ptr: "(float *)&n485". */
  /* Array #486 /._learning_rate: From_context; ptr: "(float *)0x5637b751f800". */
  /* Array #489 *.: Local_only; ptr: "(float *)&n489". */
  /* Array #496 *.: Local_only; ptr: "(float *)&n496". */
  /* Array #430 b2: From_context; ptr: "(float *)0x5637b74b37f0". */
  /* Array #492 sgd_delta_b2: Local_only; ptr: "(float *)&sgd_delta_b2". */
  /* Array #431 grad_b2: From_context; ptr: "(float *)0x5637b75ac600". */
  /* Array #494 *.: Local_only; ptr: "(float *)&n494". */
  /* Array #501 *.: Local_only; ptr: "(float *)&n501". */
  /* Array #432 b3: From_context; ptr: "(float *)0x5637b7e648d0". */
  /* Array #497 sgd_delta_b3: Local_only; ptr: "(float *)&sgd_delta_b3". */
  /* Array #433 grad_b3: From_context; ptr: "(float *)0x5637b7a9f990". */
  /* Array #499 *.: Local_only; ptr: "(float *)&n499". */
  /* Array #506 *.: Local_only; ptr: "(float *)&n506". */
  /* Array #434 w1: From_context; ptr: "(float *)0x5637b751e210". */
  /* Array #502 sgd_delta_w1: Local_only; ptr: "(float *)&sgd_delta_w1". */
  /* Array #435 grad_w1: From_context; ptr: "(float *)0x5637b751e130". */
  /* Array #504 *.: Local_only; ptr: "(float *)&n504". */
  /* Array #511 *.: Local_only; ptr: "(float *)&n511". */
  /* Array #436 w2: From_context; ptr: "(float *)0x5637b79c0220". */
  /* Array #507 sgd_delta_w2: Local_only; ptr: "(float *)&sgd_delta_w2". */
  /* Array #437 grad_w2: From_context; ptr: "(float *)0x5637b74e58b0". */
  /* Array #509 *.: Local_only; ptr: "(float *)&n509". */
  /* Array #516 *.: Local_only; ptr: "(float *)&n516". */
  /* Array #438 w3: From_context; ptr: "(float *)0x5637b75f8b80". */
  /* Array #512 sgd_delta_w3: Local_only; ptr: "(float *)&sgd_delta_w3". */
  /* Array #439 grad_w3: From_context; ptr: "(float *)0x5637b74c2cd0". */
  /* Array #514 *.: Local_only; ptr: "(float *)&n514". */
  goto scalar_loss_sgd_update;

scalar_loss_sgd_update:
  (void)fprintf (log_file, "\nCOMMENT: scalar_loss sgd update\n");
  (void)fflush (log_file);
  (void)fprintf (log_file, "\nCOMMENT: b1 param sgd step\n");
  (void)fflush (log_file);
  i708 = (int)0;
  goto loop_cond_i708;

loop_cond_i708:
  if (i708 > (int)15) goto after_loop_i708; else goto loop_body_i708;

loop_body_i708:
  (void)fprintf (log_file, "index i708 = %d\n", i708);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n491[i708] := (0.000200 * b1[i708]);\n");
  (void)fprintf (log_file, "n491[%d]{=%g} = %g = (0.0002 * b1[%d]{=%g})\n", ((double)((float *)&n491)[(i708 + (int)0 * (int)16)]), (i708 + (int)0 * (int)16), ((double)((float)0.000200 * (float *)0x5637b7e8f840[(i708 + (int)0 * (int)16)])), (i708 + (int)0 * (int)16), ((double)(float *)0x5637b7e8f840[(i708 + (int)0 * (int)16)]));
  (void)fflush (log_file);
  ((float *)&n491)[(i708 + (int)0 * (int)16)] = (float)0.000200 * (float *)0x5637b7e8f840[(i708 + (int)0 * (int)16)];
  i708 += (int)1;
  goto loop_cond_i708;

after_loop_i708:
  i710 = (int)0;
  goto loop_cond_i710;

loop_cond_i710:
  if (i710 > (int)15) goto after_loop_i710; else goto loop_body_i710;

loop_body_i710:
  (void)fprintf (log_file, "index i710 = %d\n", i710);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# sgd_delta_b1[i710] := (b1.grad[i710] + n491[i710]);\n");
  (void)fprintf (log_file, "sgd_delta_b1[%d]{=%g} = %g = (b1_grad[%d]{=%g} + n491[%d]{=%g})\n", ((double)((float *)&sgd_delta_b1)[(i710 + (int)0 * (int)16)]), (i710 + (int)0 * (int)16), ((double)((float *)0x5637b79595d0[(i710 + (int)0 * (int)16)] + ((float *)&n491)[(i710 + (int)0 * (int)16)])), (i710 + (int)0 * (int)16), ((double)(float *)0x5637b79595d0[(i710 + (int)0 * (int)16)]), (i710 + (int)0 * (int)16), ((double)((float *)&n491)[(i710 + (int)0 * (int)16)]));
  (void)fflush (log_file);
  ((float *)&sgd_delta_b1)[(i710 + (int)0 * (int)16)] = (float *)0x5637b79595d0[(i710 + (int)0 * (int)16)] + ((float *)&n491)[(i710 + (int)0 * (int)16)];
  i710 += (int)1;
  goto loop_cond_i710;

after_loop_i710:
  (void)fprintf (log_file, "# n479[0] := i626;\n");
  (void)fprintf (log_file, "n479[%d]{=%g} = %g = i626{=%d}\n", ((double)((float *)&n479)[((int)0 + (int)0 * (int)1)]), ((int)0 + (int)0 * (int)1), ((double)(float)i626), i626);
  (void)fflush (log_file);
  ((float *)&n479)[((int)0 + (int)0 * (int)1)] = (float)i626;
  (void)fprintf (log_file, "# n483[0] := (80.000000 - n479[0]);\n");
  (void)fprintf (log_file, "n483[%d]{=%g} = %g = (80. - n479[%d]{=%g})\n", ((double)((float *)&n483)[((int)0 + (int)0 * (int)1)]), ((int)0 + (int)0 * (int)1), ((double)((float)80.000000 - ((float *)&n479)[((int)0 + (int)0 * (int)1)])), ((int)0 + (int)0 * (int)1), ((double)((float *)&n479)[((int)0 + (int)0 * (int)1)]));
  (void)fflush (log_file);
  ((float *)&n483)[((int)0 + (int)0 * (int)1)] = (float)80.000000 - ((float *)&n479)[((int)0 + (int)0 * (int)1)];
  (void)fprintf (log_file, "# n485[0] := (0.100000 * n483[0]);\n");
  (void)fprintf (log_file, "n485[%d]{=%g} = %g = (0.1 * n483[%d]{=%g})\n", ((double)((float *)&n485)[((int)0 + (int)0 * (int)1)]), ((int)0 + (int)0 * (int)1), ((double)((float)0.100000 * ((float *)&n483)[((int)0 + (int)0 * (int)1)])), ((int)0 + (int)0 * (int)1), ((double)((float *)&n483)[((int)0 + (int)0 * (int)1)]));
  (void)fflush (log_file);
  ((float *)&n485)[((int)0 + (int)0 * (int)1)] = (float)0.100000 * ((float *)&n483)[((int)0 + (int)0 * (int)1)];
  (void)fprintf (log_file, "# learning_rate[0] := (n485[0] / 40.000000);\n");
  (void)fprintf (log_file, "learning_rate[%d]{=%g} = %g = (n485[%d]{=%g} / 40.)\n", ((double)(float *)0x5637b751f800[((int)0 + (int)0 * (int)1)]), ((int)0 + (int)0 * (int)1), ((double)(((float *)&n485)[((int)0 + (int)0 * (int)1)] / (float)40.000000)), ((int)0 + (int)0 * (int)1), ((double)((float *)&n485)[((int)0 + (int)0 * (int)1)]));
  (void)fflush (log_file);
  (float *)0x5637b751f800[((int)0 + (int)0 * (int)1)] = ((float *)&n485)[((int)0 + (int)0 * (int)1)] / (float)40.000000;
  i712 = (int)0;
  goto loop_cond_i712;

loop_cond_i712:
  if (i712 > (int)15) goto after_loop_i712; else goto loop_body_i712;

loop_body_i712:
  (void)fprintf (log_file, "index i712 = %d\n", i712);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n489[i712] := (learning_rate[0] * sgd_delta_b1[i712]);\n");
  (void)fprintf (log_file, "n489[%d]{=%g} = %g = (learning_rate[%d]{=%g} * sgd_delta_b1[%d]{=%g})\n", ((double)((float *)&n489)[(i712 + (int)0 * (int)16)]), (i712 + (int)0 * (int)16), ((double)((float *)0x5637b751f800[((int)0 + (int)0 * (int)1)] * ((float *)&sgd_delta_b1)[(i712 + (int)0 * (int)16)])), ((int)0 + (int)0 * (int)1), ((double)(float *)0x5637b751f800[((int)0 + (int)0 * (int)1)]), (i712 + (int)0 * (int)16), ((double)((float *)&sgd_delta_b1)[(i712 + (int)0 * (int)16)]));
  (void)fflush (log_file);
  ((float *)&n489)[(i712 + (int)0 * (int)16)] = (float *)0x5637b751f800[((int)0 + (int)0 * (int)1)] * ((float *)&sgd_delta_b1)[(i712 + (int)0 * (int)16)];
  i712 += (int)1;
  goto loop_cond_i712;

after_loop_i712:
  i714 = (int)0;
  goto loop_cond_i714;

loop_cond_i714:
  if (i714 > (int)15) goto after_loop_i714; else goto loop_body_i714;

loop_body_i714:
  (void)fprintf (log_file, "index i714 = %d\n", i714);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# b1[i714] := (b1[i714] - n489[i714]);\n");
  (void)fprintf (log_file, "b1[%d]{=%g} -= %g = n489[%d]{=%g}\n", ((double)(float *)0x5637b7e8f840[(i714 + (int)0 * (int)16)]), (i714 + (int)0 * (int)16), ((double)((float *)&n489)[(i714 + (int)0 * (int)16)]), (i714 + (int)0 * (int)16), ((double)((float *)&n489)[(i714 + (int)0 * (int)16)]));
  (void)fflush (log_file);
  (float *)0x5637b7e8f840[(i714 + (int)0 * (int)16)] -= ((float *)&n489)[(i714 + (int)0 * (int)16)];
  i714 += (int)1;
  goto loop_cond_i714;

after_loop_i714:
  (void)fprintf (log_file, "\nCOMMENT: end\n");
  (void)fflush (log_file);
  (void)fprintf (log_file, "\nCOMMENT: b2 param sgd step\n");
  (void)fflush (log_file);
  i716 = (int)0;
  goto loop_cond_i716;

loop_cond_i716:
  if (i716 > (int)15) goto after_loop_i716; else goto loop_body_i716;

loop_body_i716:
  (void)fprintf (log_file, "index i716 = %d\n", i716);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n496[i716] := (0.000200 * b2[i716]);\n");
  (void)fprintf (log_file, "n496[%d]{=%g} = %g = (0.0002 * b2[%d]{=%g})\n", ((double)((float *)&n496)[(i716 + (int)0 * (int)16)]), (i716 + (int)0 * (int)16), ((double)((float)0.000200 * (float *)0x5637b74b37f0[(i716 + (int)0 * (int)16)])), (i716 + (int)0 * (int)16), ((double)(float *)0x5637b74b37f0[(i716 + (int)0 * (int)16)]));
  (void)fflush (log_file);
  ((float *)&n496)[(i716 + (int)0 * (int)16)] = (float)0.000200 * (float *)0x5637b74b37f0[(i716 + (int)0 * (int)16)];
  i716 += (int)1;
  goto loop_cond_i716;

after_loop_i716:
  i718 = (int)0;
  goto loop_cond_i718;

loop_cond_i718:
  if (i718 > (int)15) goto after_loop_i718; else goto loop_body_i718;

loop_body_i718:
  (void)fprintf (log_file, "index i718 = %d\n", i718);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# sgd_delta_b2[i718] := (b2.grad[i718] + n496[i718]);\n");
  (void)fprintf (log_file, "sgd_delta_b2[%d]{=%g} = %g = (b2_grad[%d]{=%g} + n496[%d]{=%g})\n", ((double)((float *)&sgd_delta_b2)[(i718 + (int)0 * (int)16)]), (i718 + (int)0 * (int)16), ((double)((float *)0x5637b75ac600[(i718 + (int)0 * (int)16)] + ((float *)&n496)[(i718 + (int)0 * (int)16)])), (i718 + (int)0 * (int)16), ((double)(float *)0x5637b75ac600[(i718 + (int)0 * (int)16)]), (i718 + (int)0 * (int)16), ((double)((float *)&n496)[(i718 + (int)0 * (int)16)]));
  (void)fflush (log_file);
  ((float *)&sgd_delta_b2)[(i718 + (int)0 * (int)16)] = (float *)0x5637b75ac600[(i718 + (int)0 * (int)16)] + ((float *)&n496)[(i718 + (int)0 * (int)16)];
  i718 += (int)1;
  goto loop_cond_i718;

after_loop_i718:
  i720 = (int)0;
  goto loop_cond_i720;

loop_cond_i720:
  if (i720 > (int)15) goto after_loop_i720; else goto loop_body_i720;

loop_body_i720:
  (void)fprintf (log_file, "index i720 = %d\n", i720);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n494[i720] := (learning_rate[0] * sgd_delta_b2[i720]);\n");
  (void)fprintf (log_file, "n494[%d]{=%g} = %g = (learning_rate[%d]{=%g} * sgd_delta_b2[%d]{=%g})\n", ((double)((float *)&n494)[(i720 + (int)0 * (int)16)]), (i720 + (int)0 * (int)16), ((double)((float *)0x5637b751f800[((int)0 + (int)0 * (int)1)] * ((float *)&sgd_delta_b2)[(i720 + (int)0 * (int)16)])), ((int)0 + (int)0 * (int)1), ((double)(float *)0x5637b751f800[((int)0 + (int)0 * (int)1)]), (i720 + (int)0 * (int)16), ((double)((float *)&sgd_delta_b2)[(i720 + (int)0 * (int)16)]));
  (void)fflush (log_file);
  ((float *)&n494)[(i720 + (int)0 * (int)16)] = (float *)0x5637b751f800[((int)0 + (int)0 * (int)1)] * ((float *)&sgd_delta_b2)[(i720 + (int)0 * (int)16)];
  i720 += (int)1;
  goto loop_cond_i720;

after_loop_i720:
  i722 = (int)0;
  goto loop_cond_i722;

loop_cond_i722:
  if (i722 > (int)15) goto after_loop_i722; else goto loop_body_i722;

loop_body_i722:
  (void)fprintf (log_file, "index i722 = %d\n", i722);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# b2[i722] := (b2[i722] - n494[i722]);\n");
  (void)fprintf (log_file, "b2[%d]{=%g} -= %g = n494[%d]{=%g}\n", ((double)(float *)0x5637b74b37f0[(i722 + (int)0 * (int)16)]), (i722 + (int)0 * (int)16), ((double)((float *)&n494)[(i722 + (int)0 * (int)16)]), (i722 + (int)0 * (int)16), ((double)((float *)&n494)[(i722 + (int)0 * (int)16)]));
  (void)fflush (log_file);
  (float *)0x5637b74b37f0[(i722 + (int)0 * (int)16)] -= ((float *)&n494)[(i722 + (int)0 * (int)16)];
  i722 += (int)1;
  goto loop_cond_i722;

after_loop_i722:
  (void)fprintf (log_file, "\nCOMMENT: end\n");
  (void)fflush (log_file);
  (void)fprintf (log_file, "\nCOMMENT: b3 param sgd step\n");
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n501[0] := (0.000200 * b3[0]);\n");
  (void)fprintf (log_file, "n501[%d]{=%g} = %g = (0.0002 * b3[%d]{=%g})\n", ((double)((float *)&n501)[((int)0 + (int)0 * (int)1)]), ((int)0 + (int)0 * (int)1), ((double)((float)0.000200 * (float *)0x5637b7e648d0[((int)0 + (int)0 * (int)1)])), ((int)0 + (int)0 * (int)1), ((double)(float *)0x5637b7e648d0[((int)0 + (int)0 * (int)1)]));
  (void)fflush (log_file);
  ((float *)&n501)[((int)0 + (int)0 * (int)1)] = (float)0.000200 * (float *)0x5637b7e648d0[((int)0 + (int)0 * (int)1)];
  (void)fprintf (log_file, "# sgd_delta_b3[0] := (b3.grad[0] + n501[0]);\n");
  (void)fprintf (log_file, "sgd_delta_b3[%d]{=%g} = %g = (b3_grad[%d]{=%g} + n501[%d]{=%g})\n", ((double)((float *)&sgd_delta_b3)[((int)0 + (int)0 * (int)1)]), ((int)0 + (int)0 * (int)1), ((double)((float *)0x5637b7a9f990[((int)0 + (int)0 * (int)1)] + ((float *)&n501)[((int)0 + (int)0 * (int)1)])), ((int)0 + (int)0 * (int)1), ((double)(float *)0x5637b7a9f990[((int)0 + (int)0 * (int)1)]), ((int)0 + (int)0 * (int)1), ((double)((float *)&n501)[((int)0 + (int)0 * (int)1)]));
  (void)fflush (log_file);
  ((float *)&sgd_delta_b3)[((int)0 + (int)0 * (int)1)] = (float *)0x5637b7a9f990[((int)0 + (int)0 * (int)1)] + ((float *)&n501)[((int)0 + (int)0 * (int)1)];
  (void)fprintf (log_file, "# n499[0] := (learning_rate[0] * sgd_delta_b3[0]);\n");
  (void)fprintf (log_file, "n499[%d]{=%g} = %g = (learning_rate[%d]{=%g} * sgd_delta_b3[%d]{=%g})\n", ((double)((float *)&n499)[((int)0 + (int)0 * (int)1)]), ((int)0 + (int)0 * (int)1), ((double)((float *)0x5637b751f800[((int)0 + (int)0 * (int)1)] * ((float *)&sgd_delta_b3)[((int)0 + (int)0 * (int)1)])), ((int)0 + (int)0 * (int)1), ((double)(float *)0x5637b751f800[((int)0 + (int)0 * (int)1)]), ((int)0 + (int)0 * (int)1), ((double)((float *)&sgd_delta_b3)[((int)0 + (int)0 * (int)1)]));
  (void)fflush (log_file);
  ((float *)&n499)[((int)0 + (int)0 * (int)1)] = (float *)0x5637b751f800[((int)0 + (int)0 * (int)1)] * ((float *)&sgd_delta_b3)[((int)0 + (int)0 * (int)1)];
  (void)fprintf (log_file, "# b3[0] := (b3[0] - n499[0]);\n");
  (void)fprintf (log_file, "b3[%d]{=%g} -= %g = n499[%d]{=%g}\n", ((double)(float *)0x5637b7e648d0[((int)0 + (int)0 * (int)1)]), ((int)0 + (int)0 * (int)1), ((double)((float *)&n499)[((int)0 + (int)0 * (int)1)]), ((int)0 + (int)0 * (int)1), ((double)((float *)&n499)[((int)0 + (int)0 * (int)1)]));
  (void)fflush (log_file);
  (float *)0x5637b7e648d0[((int)0 + (int)0 * (int)1)] -= ((float *)&n499)[((int)0 + (int)0 * (int)1)];
  (void)fprintf (log_file, "\nCOMMENT: end\n");
  (void)fflush (log_file);
  (void)fprintf (log_file, "\nCOMMENT: w1 param sgd step\n");
  (void)fflush (log_file);
  i725 = (int)0;
  goto loop_cond_i725;

loop_cond_i725:
  if (i725 > (int)15) goto after_loop_i725; else goto loop_body_i725;

loop_body_i725:
  (void)fprintf (log_file, "index i725 = %d\n", i725);
  (void)fflush (log_file);
  i726 = (int)0;
  goto loop_cond_i726;

after_loop_i725:
  i729 = (int)0;
  goto loop_cond_i729;

loop_cond_i726:
  if (i726 > (int)1) goto after_loop_i726; else goto loop_body_i726;

loop_body_i726:
  (void)fprintf (log_file, "index i726 = %d\n", i726);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n506[i725, i726] := (0.000200 * w1[i725, i726]);\n");
  (void)fprintf (log_file, "n506[%d]{=%g} = %g = (0.0002 * w1[%d]{=%g})\n", ((double)((float *)&n506)[(i726 + (i725 + (int)0 * (int)16) * (int)2)]), (i726 + (i725 + (int)0 * (int)16) * (int)2), ((double)((float)0.000200 * (float *)0x5637b751e210[(i726 + (i725 + (int)0 * (int)16) * (int)2)])), (i726 + (i725 + (int)0 * (int)16) * (int)2), ((double)(float *)0x5637b751e210[(i726 + (i725 + (int)0 * (int)16) * (int)2)]));
  (void)fflush (log_file);
  ((float *)&n506)[(i726 + (i725 + (int)0 * (int)16) * (int)2)] = (float)0.000200 * (float *)0x5637b751e210[(i726 + (i725 + (int)0 * (int)16) * (int)2)];
  i726 += (int)1;
  goto loop_cond_i726;

after_loop_i726:
  i725 += (int)1;
  goto loop_cond_i725;

loop_cond_i729:
  if (i729 > (int)15) goto after_loop_i729; else goto loop_body_i729;

loop_body_i729:
  (void)fprintf (log_file, "index i729 = %d\n", i729);
  (void)fflush (log_file);
  i730 = (int)0;
  goto loop_cond_i730;

after_loop_i729:
  i733 = (int)0;
  goto loop_cond_i733;

loop_cond_i730:
  if (i730 > (int)1) goto after_loop_i730; else goto loop_body_i730;

loop_body_i730:
  (void)fprintf (log_file, "index i730 = %d\n", i730);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# sgd_delta_w1[i729, i730] := (w1.grad[i729, i730] + n506[i729, i730]);\n");
  (void)fprintf (log_file, "sgd_delta_w1[%d]{=%g} = %g = (w1_grad[%d]{=%g} + n506[%d]{=%g})\n", ((double)((float *)&sgd_delta_w1)[(i730 + (i729 + (int)0 * (int)16) * (int)2)]), (i730 + (i729 + (int)0 * (int)16) * (int)2), ((double)((float *)0x5637b751e130[(i730 + (i729 + (int)0 * (int)16) * (int)2)] + ((float *)&n506)[(i730 + (i729 + (int)0 * (int)16) * (int)2)])), (i730 + (i729 + (int)0 * (int)16) * (int)2), ((double)(float *)0x5637b751e130[(i730 + (i729 + (int)0 * (int)16) * (int)2)]), (i730 + (i729 + (int)0 * (int)16) * (int)2), ((double)((float *)&n506)[(i730 + (i729 + (int)0 * (int)16) * (int)2)]));
  (void)fflush (log_file);
  ((float *)&sgd_delta_w1)[(i730 + (i729 + (int)0 * (int)16) * (int)2)] = (float *)0x5637b751e130[(i730 + (i729 + (int)0 * (int)16) * (int)2)] + ((float *)&n506)[(i730 + (i729 + (int)0 * (int)16) * (int)2)];
  i730 += (int)1;
  goto loop_cond_i730;

after_loop_i730:
  i729 += (int)1;
  goto loop_cond_i729;

loop_cond_i733:
  if (i733 > (int)15) goto after_loop_i733; else goto loop_body_i733;

loop_body_i733:
  (void)fprintf (log_file, "index i733 = %d\n", i733);
  (void)fflush (log_file);
  i734 = (int)0;
  goto loop_cond_i734;

after_loop_i733:
  i737 = (int)0;
  goto loop_cond_i737;

loop_cond_i734:
  if (i734 > (int)1) goto after_loop_i734; else goto loop_body_i734;

loop_body_i734:
  (void)fprintf (log_file, "index i734 = %d\n", i734);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n504[i733, i734] := (learning_rate[0] * sgd_delta_w1[i733, i734]);\n");
  (void)fprintf (log_file, "n504[%d]{=%g} = %g = (learning_rate[%d]{=%g} * sgd_delta_w1[%d]{=%g})\n", ((double)((float *)&n504)[(i734 + (i733 + (int)0 * (int)16) * (int)2)]), (i734 + (i733 + (int)0 * (int)16) * (int)2), ((double)((float *)0x5637b751f800[((int)0 + (int)0 * (int)1)] * ((float *)&sgd_delta_w1)[(i734 + (i733 + (int)0 * (int)16) * (int)2)])), ((int)0 + (int)0 * (int)1), ((double)(float *)0x5637b751f800[((int)0 + (int)0 * (int)1)]), (i734 + (i733 + (int)0 * (int)16) * (int)2), ((double)((float *)&sgd_delta_w1)[(i734 + (i733 + (int)0 * (int)16) * (int)2)]));
  (void)fflush (log_file);
  ((float *)&n504)[(i734 + (i733 + (int)0 * (int)16) * (int)2)] = (float *)0x5637b751f800[((int)0 + (int)0 * (int)1)] * ((float *)&sgd_delta_w1)[(i734 + (i733 + (int)0 * (int)16) * (int)2)];
  i734 += (int)1;
  goto loop_cond_i734;

after_loop_i734:
  i733 += (int)1;
  goto loop_cond_i733;

loop_cond_i737:
  if (i737 > (int)15) goto after_loop_i737; else goto loop_body_i737;

loop_body_i737:
  (void)fprintf (log_file, "index i737 = %d\n", i737);
  (void)fflush (log_file);
  i738 = (int)0;
  goto loop_cond_i738;

after_loop_i737:
  (void)fprintf (log_file, "\nCOMMENT: end\n");
  (void)fflush (log_file);
  (void)fprintf (log_file, "\nCOMMENT: w2 param sgd step\n");
  (void)fflush (log_file);
  i741 = (int)0;
  goto loop_cond_i741;

loop_cond_i738:
  if (i738 > (int)1) goto after_loop_i738; else goto loop_body_i738;

loop_body_i738:
  (void)fprintf (log_file, "index i738 = %d\n", i738);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# w1[i737, i738] := (w1[i737, i738] - n504[i737, i738]);\n");
  (void)fprintf (log_file, "w1[%d]{=%g} -= %g = n504[%d]{=%g}\n", ((double)(float *)0x5637b751e210[(i738 + (i737 + (int)0 * (int)16) * (int)2)]), (i738 + (i737 + (int)0 * (int)16) * (int)2), ((double)((float *)&n504)[(i738 + (i737 + (int)0 * (int)16) * (int)2)]), (i738 + (i737 + (int)0 * (int)16) * (int)2), ((double)((float *)&n504)[(i738 + (i737 + (int)0 * (int)16) * (int)2)]));
  (void)fflush (log_file);
  (float *)0x5637b751e210[(i738 + (i737 + (int)0 * (int)16) * (int)2)] -= ((float *)&n504)[(i738 + (i737 + (int)0 * (int)16) * (int)2)];
  i738 += (int)1;
  goto loop_cond_i738;

after_loop_i738:
  i737 += (int)1;
  goto loop_cond_i737;

loop_cond_i741:
  if (i741 > (int)15) goto after_loop_i741; else goto loop_body_i741;

loop_body_i741:
  (void)fprintf (log_file, "index i741 = %d\n", i741);
  (void)fflush (log_file);
  i742 = (int)0;
  goto loop_cond_i742;

after_loop_i741:
  i745 = (int)0;
  goto loop_cond_i745;

loop_cond_i742:
  if (i742 > (int)15) goto after_loop_i742; else goto loop_body_i742;

loop_body_i742:
  (void)fprintf (log_file, "index i742 = %d\n", i742);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n511[i741, i742] := (0.000200 * w2[i741, i742]);\n");
  (void)fprintf (log_file, "n511[%d]{=%g} = %g = (0.0002 * w2[%d]{=%g})\n", ((double)((float *)&n511)[(i742 + (i741 + (int)0 * (int)16) * (int)16)]), (i742 + (i741 + (int)0 * (int)16) * (int)16), ((double)((float)0.000200 * (float *)0x5637b79c0220[(i742 + (i741 + (int)0 * (int)16) * (int)16)])), (i742 + (i741 + (int)0 * (int)16) * (int)16), ((double)(float *)0x5637b79c0220[(i742 + (i741 + (int)0 * (int)16) * (int)16)]));
  (void)fflush (log_file);
  ((float *)&n511)[(i742 + (i741 + (int)0 * (int)16) * (int)16)] = (float)0.000200 * (float *)0x5637b79c0220[(i742 + (i741 + (int)0 * (int)16) * (int)16)];
  i742 += (int)1;
  goto loop_cond_i742;

after_loop_i742:
  i741 += (int)1;
  goto loop_cond_i741;

loop_cond_i745:
  if (i745 > (int)15) goto after_loop_i745; else goto loop_body_i745;

loop_body_i745:
  (void)fprintf (log_file, "index i745 = %d\n", i745);
  (void)fflush (log_file);
  i746 = (int)0;
  goto loop_cond_i746;

after_loop_i745:
  i749 = (int)0;
  goto loop_cond_i749;

loop_cond_i746:
  if (i746 > (int)15) goto after_loop_i746; else goto loop_body_i746;

loop_body_i746:
  (void)fprintf (log_file, "index i746 = %d\n", i746);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# sgd_delta_w2[i745, i746] := (w2.grad[i745, i746] + n511[i745, i746]);\n");
  (void)fprintf (log_file, "sgd_delta_w2[%d]{=%g} = %g = (w2_grad[%d]{=%g} + n511[%d]{=%g})\n", ((double)((float *)&sgd_delta_w2)[(i746 + (i745 + (int)0 * (int)16) * (int)16)]), (i746 + (i745 + (int)0 * (int)16) * (int)16), ((double)((float *)0x5637b74e58b0[(i746 + (i745 + (int)0 * (int)16) * (int)16)] + ((float *)&n511)[(i746 + (i745 + (int)0 * (int)16) * (int)16)])), (i746 + (i745 + (int)0 * (int)16) * (int)16), ((double)(float *)0x5637b74e58b0[(i746 + (i745 + (int)0 * (int)16) * (int)16)]), (i746 + (i745 + (int)0 * (int)16) * (int)16), ((double)((float *)&n511)[(i746 + (i745 + (int)0 * (int)16) * (int)16)]));
  (void)fflush (log_file);
  ((float *)&sgd_delta_w2)[(i746 + (i745 + (int)0 * (int)16) * (int)16)] = (float *)0x5637b74e58b0[(i746 + (i745 + (int)0 * (int)16) * (int)16)] + ((float *)&n511)[(i746 + (i745 + (int)0 * (int)16) * (int)16)];
  i746 += (int)1;
  goto loop_cond_i746;

after_loop_i746:
  i745 += (int)1;
  goto loop_cond_i745;

loop_cond_i749:
  if (i749 > (int)15) goto after_loop_i749; else goto loop_body_i749;

loop_body_i749:
  (void)fprintf (log_file, "index i749 = %d\n", i749);
  (void)fflush (log_file);
  i750 = (int)0;
  goto loop_cond_i750;

after_loop_i749:
  i753 = (int)0;
  goto loop_cond_i753;

loop_cond_i750:
  if (i750 > (int)15) goto after_loop_i750; else goto loop_body_i750;

loop_body_i750:
  (void)fprintf (log_file, "index i750 = %d\n", i750);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n509[i749, i750] := (learning_rate[0] * sgd_delta_w2[i749, i750]);\n");
  (void)fprintf (log_file, "n509[%d]{=%g} = %g = (learning_rate[%d]{=%g} * sgd_delta_w2[%d]{=%g})\n", ((double)((float *)&n509)[(i750 + (i749 + (int)0 * (int)16) * (int)16)]), (i750 + (i749 + (int)0 * (int)16) * (int)16), ((double)((float *)0x5637b751f800[((int)0 + (int)0 * (int)1)] * ((float *)&sgd_delta_w2)[(i750 + (i749 + (int)0 * (int)16) * (int)16)])), ((int)0 + (int)0 * (int)1), ((double)(float *)0x5637b751f800[((int)0 + (int)0 * (int)1)]), (i750 + (i749 + (int)0 * (int)16) * (int)16), ((double)((float *)&sgd_delta_w2)[(i750 + (i749 + (int)0 * (int)16) * (int)16)]));
  (void)fflush (log_file);
  ((float *)&n509)[(i750 + (i749 + (int)0 * (int)16) * (int)16)] = (float *)0x5637b751f800[((int)0 + (int)0 * (int)1)] * ((float *)&sgd_delta_w2)[(i750 + (i749 + (int)0 * (int)16) * (int)16)];
  i750 += (int)1;
  goto loop_cond_i750;

after_loop_i750:
  i749 += (int)1;
  goto loop_cond_i749;

loop_cond_i753:
  if (i753 > (int)15) goto after_loop_i753; else goto loop_body_i753;

loop_body_i753:
  (void)fprintf (log_file, "index i753 = %d\n", i753);
  (void)fflush (log_file);
  i754 = (int)0;
  goto loop_cond_i754;

after_loop_i753:
  (void)fprintf (log_file, "\nCOMMENT: end\n");
  (void)fflush (log_file);
  (void)fprintf (log_file, "\nCOMMENT: w3 param sgd step\n");
  (void)fflush (log_file);
  i756 = (int)0;
  goto loop_cond_i756;

loop_cond_i754:
  if (i754 > (int)15) goto after_loop_i754; else goto loop_body_i754;

loop_body_i754:
  (void)fprintf (log_file, "index i754 = %d\n", i754);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# w2[i753, i754] := (w2[i753, i754] - n509[i753, i754]);\n");
  (void)fprintf (log_file, "w2[%d]{=%g} -= %g = n509[%d]{=%g}\n", ((double)(float *)0x5637b79c0220[(i754 + (i753 + (int)0 * (int)16) * (int)16)]), (i754 + (i753 + (int)0 * (int)16) * (int)16), ((double)((float *)&n509)[(i754 + (i753 + (int)0 * (int)16) * (int)16)]), (i754 + (i753 + (int)0 * (int)16) * (int)16), ((double)((float *)&n509)[(i754 + (i753 + (int)0 * (int)16) * (int)16)]));
  (void)fflush (log_file);
  (float *)0x5637b79c0220[(i754 + (i753 + (int)0 * (int)16) * (int)16)] -= ((float *)&n509)[(i754 + (i753 + (int)0 * (int)16) * (int)16)];
  i754 += (int)1;
  goto loop_cond_i754;

after_loop_i754:
  i753 += (int)1;
  goto loop_cond_i753;

loop_cond_i756:
  if (i756 > (int)15) goto after_loop_i756; else goto loop_body_i756;

loop_body_i756:
  (void)fprintf (log_file, "index i756 = %d\n", i756);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n516[0, i756] := (0.000200 * w3[0, i756]);\n");
  (void)fprintf (log_file, "n516[%d]{=%g} = %g = (0.0002 * w3[%d]{=%g})\n", ((double)((float *)&n516)[(i756 + ((int)0 + (int)0 * (int)1) * (int)16)]), (i756 + ((int)0 + (int)0 * (int)1) * (int)16), ((double)((float)0.000200 * (float *)0x5637b75f8b80[(i756 + ((int)0 + (int)0 * (int)1) * (int)16)])), (i756 + ((int)0 + (int)0 * (int)1) * (int)16), ((double)(float *)0x5637b75f8b80[(i756 + ((int)0 + (int)0 * (int)1) * (int)16)]));
  (void)fflush (log_file);
  ((float *)&n516)[(i756 + ((int)0 + (int)0 * (int)1) * (int)16)] = (float)0.000200 * (float *)0x5637b75f8b80[(i756 + ((int)0 + (int)0 * (int)1) * (int)16)];
  i756 += (int)1;
  goto loop_cond_i756;

after_loop_i756:
  i758 = (int)0;
  goto loop_cond_i758;

loop_cond_i758:
  if (i758 > (int)15) goto after_loop_i758; else goto loop_body_i758;

loop_body_i758:
  (void)fprintf (log_file, "index i758 = %d\n", i758);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# sgd_delta_w3[0, i758] := (w3.grad[0, i758] + n516[0, i758]);\n");
  (void)fprintf (log_file, "sgd_delta_w3[%d]{=%g} = %g = (w3_grad[%d]{=%g} + n516[%d]{=%g})\n", ((double)((float *)&sgd_delta_w3)[(i758 + ((int)0 + (int)0 * (int)1) * (int)16)]), (i758 + ((int)0 + (int)0 * (int)1) * (int)16), ((double)((float *)0x5637b74c2cd0[(i758 + ((int)0 + (int)0 * (int)1) * (int)16)] + ((float *)&n516)[(i758 + ((int)0 + (int)0 * (int)1) * (int)16)])), (i758 + ((int)0 + (int)0 * (int)1) * (int)16), ((double)(float *)0x5637b74c2cd0[(i758 + ((int)0 + (int)0 * (int)1) * (int)16)]), (i758 + ((int)0 + (int)0 * (int)1) * (int)16), ((double)((float *)&n516)[(i758 + ((int)0 + (int)0 * (int)1) * (int)16)]));
  (void)fflush (log_file);
  ((float *)&sgd_delta_w3)[(i758 + ((int)0 + (int)0 * (int)1) * (int)16)] = (float *)0x5637b74c2cd0[(i758 + ((int)0 + (int)0 * (int)1) * (int)16)] + ((float *)&n516)[(i758 + ((int)0 + (int)0 * (int)1) * (int)16)];
  i758 += (int)1;
  goto loop_cond_i758;

after_loop_i758:
  i760 = (int)0;
  goto loop_cond_i760;

loop_cond_i760:
  if (i760 > (int)15) goto after_loop_i760; else goto loop_body_i760;

loop_body_i760:
  (void)fprintf (log_file, "index i760 = %d\n", i760);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n514[0, i760] := (learning_rate[0] * sgd_delta_w3[0, i760]);\n");
  (void)fprintf (log_file, "n514[%d]{=%g} = %g = (learning_rate[%d]{=%g} * sgd_delta_w3[%d]{=%g})\n", ((double)((float *)&n514)[(i760 + ((int)0 + (int)0 * (int)1) * (int)16)]), (i760 + ((int)0 + (int)0 * (int)1) * (int)16), ((double)((float *)0x5637b751f800[((int)0 + (int)0 * (int)1)] * ((float *)&sgd_delta_w3)[(i760 + ((int)0 + (int)0 * (int)1) * (int)16)])), ((int)0 + (int)0 * (int)1), ((double)(float *)0x5637b751f800[((int)0 + (int)0 * (int)1)]), (i760 + ((int)0 + (int)0 * (int)1) * (int)16), ((double)((float *)&sgd_delta_w3)[(i760 + ((int)0 + (int)0 * (int)1) * (int)16)]));
  (void)fflush (log_file);
  ((float *)&n514)[(i760 + ((int)0 + (int)0 * (int)1) * (int)16)] = (float *)0x5637b751f800[((int)0 + (int)0 * (int)1)] * ((float *)&sgd_delta_w3)[(i760 + ((int)0 + (int)0 * (int)1) * (int)16)];
  i760 += (int)1;
  goto loop_cond_i760;

after_loop_i760:
  i762 = (int)0;
  goto loop_cond_i762;

loop_cond_i762:
  if (i762 > (int)15) goto after_loop_i762; else goto loop_body_i762;

loop_body_i762:
  (void)fprintf (log_file, "index i762 = %d\n", i762);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# w3[0, i762] := (w3[0, i762] - n514[0, i762]);\n");
  (void)fprintf (log_file, "w3[%d]{=%g} -= %g = n514[%d]{=%g}\n", ((double)(float *)0x5637b75f8b80[(i762 + ((int)0 + (int)0 * (int)1) * (int)16)]), (i762 + ((int)0 + (int)0 * (int)1) * (int)16), ((double)((float *)&n514)[(i762 + ((int)0 + (int)0 * (int)1) * (int)16)]), (i762 + ((int)0 + (int)0 * (int)1) * (int)16), ((double)((float *)&n514)[(i762 + ((int)0 + (int)0 * (int)1) * (int)16)]));
  (void)fflush (log_file);
  (float *)0x5637b75f8b80[(i762 + ((int)0 + (int)0 * (int)1) * (int)16)] -= ((float *)&n514)[(i762 + ((int)0 + (int)0 * (int)1) * (int)16)];
  i762 += (int)1;
  goto loop_cond_i762;

after_loop_i762:
  (void)fprintf (log_file, "\nCOMMENT: end\n");
  (void)fflush (log_file);
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

