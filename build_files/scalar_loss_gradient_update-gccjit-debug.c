extern void
scalar_loss_gradient_update (float * w1_grad, float * b1_grad, float * w2_grad, float * b2_grad, float * w3_grad, float * b3_grad, float * scalar_loss, float * b3, float * w3, float * b2, float * w2, float * b1, float * w1, const char * log_file_name, int i625, int i626)
{
  FILE * log_file;
  float[60] expectation;
  float[120] input;
  float[960] n444;
  float[960] n446;
  float[960] n448;
  float[960] n450;
  float[960] n452;
  float[960] n454;
  float[60] n456;
  float[60] mlp;
  float[60] n460;
  float[60] n463;
  float[60] n465;
  float[1] n468;
  float[960] n444_grad;
  float[960] n446_grad;
  float[960] n448_grad;
  float[960] n450_grad;
  float[960] n452_grad;
  float[960] n454_grad;
  float[60] n456_grad;
  float[60] mlp_grad;
  float[60] n460_grad;
  float[60] n463_grad;
  float[60] n465_grad;
  float[1] n475;
  int i628;
  int i631;
  int i632;
  int i636;
  int i637;
  int i638;
  int i641;
  int i642;
  int i645;
  int i646;
  int i650;
  int i651;
  int i652;
  int i655;
  int i656;
  int i659;
  int i660;
  int i663;
  int i664;
  int i666;
  int i668;
  int i670;
  int i672;
  int i675;
  int i676;
  int i677;
  int i678;
  int i679;
  int i680;
  int i681;
  int i682;
  int i683;
  int i684;
  int i685;
  int i686;
  int i687;
  int i688;
  int i689;
  int i690;
  int i691;
  int i692;
  int i693;
  int i694;
  int i695;
  int i696;
  int i697;
  int i698;
  int i699;
  int i700;
  int i701;
  int i702;
  int i703;
  int i704;
  int i705;
  int i706;

init_scalar_loss_gradient_update:
  log_file = fopen (log_file_name, "w");
  (void)fprintf (log_file, "index i625 = %d\n", i625);
  (void)fflush (log_file);
  (void)fprintf (log_file, "index i626 = %d\n", i626);
  (void)fflush (log_file);
  /* Array #443 @|_expectation: Local_only; ptr: "(float *)&expectation". */
  /* Array #441 moons_classes: Constant_from_host; ptr: "(float *)0x5637b74df300". */
  /* Array #442 @|_input: Local_only; ptr: "(float *)&input". */
  /* Array #440 moons_flat: Constant_from_host; ptr: "(float *)0x5637b7e42cd0". */
  /* Array #444 *: Local_only; ptr: "(float *)&n444". */
  (void)fprintf (log_file, "memset_zero(n444) where before first element = %g\n", ((double)((float *)&n444)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n444), (int)0, (size_t)3840);
  /* Array #434 w1: From_context; ptr: w1. */
  /* Array #446 +: Local_only; ptr: "(float *)&n446". */
  /* Array #428 b1: From_context; ptr: b1. */
  /* Array #448 ?/: Local_only; ptr: "(float *)&n448". */
  /* Array #450 *: Local_only; ptr: "(float *)&n450". */
  (void)fprintf (log_file, "memset_zero(n450) where before first element = %g\n", ((double)((float *)&n450)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n450), (int)0, (size_t)3840);
  /* Array #436 w2: From_context; ptr: w2. */
  /* Array #452 +: Local_only; ptr: "(float *)&n452". */
  /* Array #430 b2: From_context; ptr: b2. */
  /* Array #454 ?/: Local_only; ptr: "(float *)&n454". */
  /* Array #456 *: Local_only; ptr: "(float *)&n456". */
  (void)fprintf (log_file, "memset_zero(n456) where before first element = %g\n", ((double)((float *)&n456)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n456), (int)0, (size_t)240);
  /* Array #438 w3: From_context; ptr: w3. */
  /* Array #458 +_mlp: Local_only; ptr: "(float *)&mlp". */
  /* Array #432 b3: From_context; ptr: b3. */
  /* Array #460 *.: Local_only; ptr: "(float *)&n460". */
  /* Array #463 -: Local_only; ptr: "(float *)&n463". */
  /* Array #465 ?/: Local_only; ptr: "(float *)&n465". */
  /* Array #468 =>: Local_only; ptr: "(float *)&n468". */
  (void)fprintf (log_file, "memset_zero(n468) where before first element = %g\n", ((double)((float *)&n468)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n468), (int)0, (size_t)4);
  /* Array #470 /._scalar_loss: From_context; ptr: scalar_loss. */
  /* Array #433 grad_b3: From_context; ptr: b3_grad. */
  (void)fprintf (log_file, "memset_zero(b3_grad) where before first element = %g\n", ((double)b3_grad[(int)0]));
  (void)fflush (log_file);
  (void)memset (b3_grad, (int)0, (size_t)4);
  /* Array #439 grad_w3: From_context; ptr: w3_grad. */
  (void)fprintf (log_file, "memset_zero(w3_grad) where before first element = %g\n", ((double)w3_grad[(int)0]));
  (void)fflush (log_file);
  (void)memset (w3_grad, (int)0, (size_t)64);
  /* Array #431 grad_b2: From_context; ptr: b2_grad. */
  (void)fprintf (log_file, "memset_zero(b2_grad) where before first element = %g\n", ((double)b2_grad[(int)0]));
  (void)fflush (log_file);
  (void)memset (b2_grad, (int)0, (size_t)64);
  /* Array #437 grad_w2: From_context; ptr: w2_grad. */
  (void)fprintf (log_file, "memset_zero(w2_grad) where before first element = %g\n", ((double)w2_grad[(int)0]));
  (void)fflush (log_file);
  (void)memset (w2_grad, (int)0, (size_t)1024);
  /* Array #429 grad_b1: From_context; ptr: b1_grad. */
  (void)fprintf (log_file, "memset_zero(b1_grad) where before first element = %g\n", ((double)b1_grad[(int)0]));
  (void)fflush (log_file);
  (void)memset (b1_grad, (int)0, (size_t)64);
  /* Array #435 grad_w1: From_context; ptr: w1_grad. */
  (void)fprintf (log_file, "memset_zero(w1_grad) where before first element = %g\n", ((double)w1_grad[(int)0]));
  (void)fflush (log_file);
  (void)memset (w1_grad, (int)0, (size_t)128);
  /* Array #445 grad_*: Local_only; ptr: "(float *)&n444_grad". */
  (void)fprintf (log_file, "memset_zero(n444_grad) where before first element = %g\n", ((double)((float *)&n444_grad)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n444_grad), (int)0, (size_t)3840);
  /* Array #447 grad_+: Local_only; ptr: "(float *)&n446_grad". */
  (void)fprintf (log_file, "memset_zero(n446_grad) where before first element = %g\n", ((double)((float *)&n446_grad)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n446_grad), (int)0, (size_t)3840);
  /* Array #449 grad_?/: Local_only; ptr: "(float *)&n448_grad". */
  (void)fprintf (log_file, "memset_zero(n448_grad) where before first element = %g\n", ((double)((float *)&n448_grad)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n448_grad), (int)0, (size_t)3840);
  /* Array #451 grad_*: Local_only; ptr: "(float *)&n450_grad". */
  (void)fprintf (log_file, "memset_zero(n450_grad) where before first element = %g\n", ((double)((float *)&n450_grad)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n450_grad), (int)0, (size_t)3840);
  /* Array #453 grad_+: Local_only; ptr: "(float *)&n452_grad". */
  (void)fprintf (log_file, "memset_zero(n452_grad) where before first element = %g\n", ((double)((float *)&n452_grad)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n452_grad), (int)0, (size_t)3840);
  /* Array #455 grad_?/: Local_only; ptr: "(float *)&n454_grad". */
  (void)fprintf (log_file, "memset_zero(n454_grad) where before first element = %g\n", ((double)((float *)&n454_grad)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n454_grad), (int)0, (size_t)3840);
  /* Array #457 grad_*: Local_only; ptr: "(float *)&n456_grad". */
  (void)fprintf (log_file, "memset_zero(n456_grad) where before first element = %g\n", ((double)((float *)&n456_grad)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n456_grad), (int)0, (size_t)240);
  /* Array #459 grad_+_mlp: Local_only; ptr: "(float *)&mlp_grad". */
  (void)fprintf (log_file, "memset_zero(mlp_grad) where before first element = %g\n", ((double)((float *)&mlp_grad)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&mlp_grad), (int)0, (size_t)240);
  /* Array #461 grad_*.: Local_only; ptr: "(float *)&n460_grad". */
  (void)fprintf (log_file, "memset_zero(n460_grad) where before first element = %g\n", ((double)((float *)&n460_grad)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n460_grad), (int)0, (size_t)240);
  /* Array #464 grad_-: Local_only; ptr: "(float *)&n463_grad". */
  (void)fprintf (log_file, "memset_zero(n463_grad) where before first element = %g\n", ((double)((float *)&n463_grad)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n463_grad), (int)0, (size_t)240);
  /* Array #466 grad_?/: Local_only; ptr: "(float *)&n465_grad". */
  (void)fprintf (log_file, "memset_zero(n465_grad) where before first element = %g\n", ((double)((float *)&n465_grad)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n465_grad), (int)0, (size_t)240);
  /* Array #475 *.: Local_only; ptr: "(float *)&n475". */
  goto scalar_loss_gradient_update;

scalar_loss_gradient_update:
  (void)fprintf (log_file, "\nCOMMENT: scalar_loss gradient update\n");
  (void)fflush (log_file);
  (void)fprintf (log_file, "\nCOMMENT: scalar_loss fwd\n");
  (void)fflush (log_file);
  i628 = (int)0;
  goto loop_cond_i628;

loop_cond_i628:
  if (i628 > (int)59) goto after_loop_i628; else goto loop_body_i628;

loop_body_i628:
  (void)fprintf (log_file, "index i628 = %d\n", i628);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# expectation[i628, 0] := moons_classes[i625, i628, 0];\n");
  (void)fprintf (log_file, "expectation[%d]{=%g} = %g = moons_classes[%d]{=%g}\n", ((double)((float *)&expectation)[((int)0 + (i628 + (int)0 * (int)60) * (int)1)]), ((int)0 + (i628 + (int)0 * (int)60) * (int)1), ((double)(float *)0x5637b74df300[((int)0 + (i628 + (i625 + (int)0 * (int)40) * (int)60) * (int)1)]), ((int)0 + (i628 + (i625 + (int)0 * (int)40) * (int)60) * (int)1), ((double)(float *)0x5637b74df300[((int)0 + (i628 + (i625 + (int)0 * (int)40) * (int)60) * (int)1)]));
  (void)fflush (log_file);
  ((float *)&expectation)[((int)0 + (i628 + (int)0 * (int)60) * (int)1)] = (float *)0x5637b74df300[((int)0 + (i628 + (i625 + (int)0 * (int)40) * (int)60) * (int)1)];
  i628 += (int)1;
  goto loop_cond_i628;

after_loop_i628:
  i631 = (int)0;
  goto loop_cond_i631;

loop_cond_i631:
  if (i631 > (int)59) goto after_loop_i631; else goto loop_body_i631;

loop_body_i631:
  (void)fprintf (log_file, "index i631 = %d\n", i631);
  (void)fflush (log_file);
  i632 = (int)0;
  goto loop_cond_i632;

after_loop_i631:
  (void)fprintf (log_file, "memset_zero(n444) where before first element = %g\n", ((double)((float *)&n444)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n444), (int)0, (size_t)3840);
  i636 = (int)0;
  goto loop_cond_i636;

loop_cond_i632:
  if (i632 > (int)1) goto after_loop_i632; else goto loop_body_i632;

loop_body_i632:
  (void)fprintf (log_file, "index i632 = %d\n", i632);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# input[i631, i632] := moons_flat[i625, i631, i632];\n");
  (void)fprintf (log_file, "input[%d]{=%g} = %g = moons_flat[%d]{=%g}\n", ((double)((float *)&input)[(i632 + (i631 + (int)0 * (int)60) * (int)2)]), (i632 + (i631 + (int)0 * (int)60) * (int)2), ((double)(float *)0x5637b7e42cd0[(i632 + (i631 + (i625 + (int)0 * (int)40) * (int)60) * (int)2)]), (i632 + (i631 + (i625 + (int)0 * (int)40) * (int)60) * (int)2), ((double)(float *)0x5637b7e42cd0[(i632 + (i631 + (i625 + (int)0 * (int)40) * (int)60) * (int)2)]));
  (void)fflush (log_file);
  ((float *)&input)[(i632 + (i631 + (int)0 * (int)60) * (int)2)] = (float *)0x5637b7e42cd0[(i632 + (i631 + (i625 + (int)0 * (int)40) * (int)60) * (int)2)];
  i632 += (int)1;
  goto loop_cond_i632;

after_loop_i632:
  i631 += (int)1;
  goto loop_cond_i631;

loop_cond_i636:
  if (i636 > (int)59) goto after_loop_i636; else goto loop_body_i636;

loop_body_i636:
  (void)fprintf (log_file, "index i636 = %d\n", i636);
  (void)fflush (log_file);
  i637 = (int)0;
  goto loop_cond_i637;

after_loop_i636:
  i641 = (int)0;
  goto loop_cond_i641;

loop_cond_i637:
  if (i637 > (int)15) goto after_loop_i637; else goto loop_body_i637;

loop_body_i637:
  (void)fprintf (log_file, "index i637 = %d\n", i637);
  (void)fflush (log_file);
  i638 = (int)0;
  goto loop_cond_i638;

after_loop_i637:
  i636 += (int)1;
  goto loop_cond_i636;

loop_cond_i638:
  if (i638 > (int)1) goto after_loop_i638; else goto loop_body_i638;

loop_body_i638:
  (void)fprintf (log_file, "index i638 = %d\n", i638);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n444[i636, i637] :=$  (n444[i636, i637] + (w1[i637, i638] * input[i636, i638]));\n");
  (void)fprintf (log_file, "n444[%d]{=%g} += %g = (w1[%d]{=%g} * input[%d]{=%g})\n", ((double)((float *)&n444)[(i637 + (i636 + (int)0 * (int)60) * (int)16)]), (i637 + (i636 + (int)0 * (int)60) * (int)16), ((double)(w1[(i638 + (i637 + (int)0 * (int)16) * (int)2)] * ((float *)&input)[(i638 + (i636 + (int)0 * (int)60) * (int)2)])), (i638 + (i637 + (int)0 * (int)16) * (int)2), ((double)w1[(i638 + (i637 + (int)0 * (int)16) * (int)2)]), (i638 + (i636 + (int)0 * (int)60) * (int)2), ((double)((float *)&input)[(i638 + (i636 + (int)0 * (int)60) * (int)2)]));
  (void)fflush (log_file);
  ((float *)&n444)[(i637 + (i636 + (int)0 * (int)60) * (int)16)] += w1[(i638 + (i637 + (int)0 * (int)16) * (int)2)] * ((float *)&input)[(i638 + (i636 + (int)0 * (int)60) * (int)2)];
  i638 += (int)1;
  goto loop_cond_i638;

after_loop_i638:
  i637 += (int)1;
  goto loop_cond_i637;

loop_cond_i641:
  if (i641 > (int)59) goto after_loop_i641; else goto loop_body_i641;

loop_body_i641:
  (void)fprintf (log_file, "index i641 = %d\n", i641);
  (void)fflush (log_file);
  i642 = (int)0;
  goto loop_cond_i642;

after_loop_i641:
  i645 = (int)0;
  goto loop_cond_i645;

loop_cond_i642:
  if (i642 > (int)15) goto after_loop_i642; else goto loop_body_i642;

loop_body_i642:
  (void)fprintf (log_file, "index i642 = %d\n", i642);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n446[i641, i642] := (b1[i642] + n444[i641, i642]);\n");
  (void)fprintf (log_file, "n446[%d]{=%g} = %g = (b1[%d]{=%g} + n444[%d]{=%g})\n", ((double)((float *)&n446)[(i642 + (i641 + (int)0 * (int)60) * (int)16)]), (i642 + (i641 + (int)0 * (int)60) * (int)16), ((double)(b1[(i642 + (int)0 * (int)16)] + ((float *)&n444)[(i642 + (i641 + (int)0 * (int)60) * (int)16)])), (i642 + (int)0 * (int)16), ((double)b1[(i642 + (int)0 * (int)16)]), (i642 + (i641 + (int)0 * (int)60) * (int)16), ((double)((float *)&n444)[(i642 + (i641 + (int)0 * (int)60) * (int)16)]));
  (void)fflush (log_file);
  ((float *)&n446)[(i642 + (i641 + (int)0 * (int)60) * (int)16)] = b1[(i642 + (int)0 * (int)16)] + ((float *)&n444)[(i642 + (i641 + (int)0 * (int)60) * (int)16)];
  i642 += (int)1;
  goto loop_cond_i642;

after_loop_i642:
  i641 += (int)1;
  goto loop_cond_i641;

loop_cond_i645:
  if (i645 > (int)59) goto after_loop_i645; else goto loop_body_i645;

loop_body_i645:
  (void)fprintf (log_file, "index i645 = %d\n", i645);
  (void)fflush (log_file);
  i646 = (int)0;
  goto loop_cond_i646;

after_loop_i645:
  (void)fprintf (log_file, "memset_zero(n450) where before first element = %g\n", ((double)((float *)&n450)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n450), (int)0, (size_t)3840);
  i650 = (int)0;
  goto loop_cond_i650;

loop_cond_i646:
  if (i646 > (int)15) goto after_loop_i646; else goto loop_body_i646;

loop_body_i646:
  (void)fprintf (log_file, "index i646 = %d\n", i646);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n448[i645, i646] := relu(n446[i645, i646]);\n");
  (void)fprintf (log_file, "n448[%d]{=%g} = %g = (n446[%d]{=%g} > 0.0 ? n446[%d]{=%g} : 0.0)\n", ((double)((float *)&n448)[(i646 + (i645 + (int)0 * (int)60) * (int)16)]), (i646 + (i645 + (int)0 * (int)60) * (int)16), ((double)((float)(int)((float)0 < ((float *)&n446)[(i646 + (i645 + (int)0 * (int)60) * (int)16)]) * ((float *)&n446)[(i646 + (i645 + (int)0 * (int)60) * (int)16)])), (i646 + (i645 + (int)0 * (int)60) * (int)16), ((double)((float *)&n446)[(i646 + (i645 + (int)0 * (int)60) * (int)16)]), (i646 + (i645 + (int)0 * (int)60) * (int)16), ((double)((float *)&n446)[(i646 + (i645 + (int)0 * (int)60) * (int)16)]));
  (void)fflush (log_file);
  ((float *)&n448)[(i646 + (i645 + (int)0 * (int)60) * (int)16)] = (float)(int)((float)0 < ((float *)&n446)[(i646 + (i645 + (int)0 * (int)60) * (int)16)]) * ((float *)&n446)[(i646 + (i645 + (int)0 * (int)60) * (int)16)];
  i646 += (int)1;
  goto loop_cond_i646;

after_loop_i646:
  i645 += (int)1;
  goto loop_cond_i645;

loop_cond_i650:
  if (i650 > (int)59) goto after_loop_i650; else goto loop_body_i650;

loop_body_i650:
  (void)fprintf (log_file, "index i650 = %d\n", i650);
  (void)fflush (log_file);
  i651 = (int)0;
  goto loop_cond_i651;

after_loop_i650:
  i655 = (int)0;
  goto loop_cond_i655;

loop_cond_i651:
  if (i651 > (int)15) goto after_loop_i651; else goto loop_body_i651;

loop_body_i651:
  (void)fprintf (log_file, "index i651 = %d\n", i651);
  (void)fflush (log_file);
  i652 = (int)0;
  goto loop_cond_i652;

after_loop_i651:
  i650 += (int)1;
  goto loop_cond_i650;

loop_cond_i652:
  if (i652 > (int)15) goto after_loop_i652; else goto loop_body_i652;

loop_body_i652:
  (void)fprintf (log_file, "index i652 = %d\n", i652);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n450[i650, i651] := (n450[i650, i651] + (w2[i651, i652] * n448[i650, i652]));\n");
  (void)fprintf (log_file, "n450[%d]{=%g} += %g = (w2[%d]{=%g} * n448[%d]{=%g})\n", ((double)((float *)&n450)[(i651 + (i650 + (int)0 * (int)60) * (int)16)]), (i651 + (i650 + (int)0 * (int)60) * (int)16), ((double)(w2[(i652 + (i651 + (int)0 * (int)16) * (int)16)] * ((float *)&n448)[(i652 + (i650 + (int)0 * (int)60) * (int)16)])), (i652 + (i651 + (int)0 * (int)16) * (int)16), ((double)w2[(i652 + (i651 + (int)0 * (int)16) * (int)16)]), (i652 + (i650 + (int)0 * (int)60) * (int)16), ((double)((float *)&n448)[(i652 + (i650 + (int)0 * (int)60) * (int)16)]));
  (void)fflush (log_file);
  ((float *)&n450)[(i651 + (i650 + (int)0 * (int)60) * (int)16)] += w2[(i652 + (i651 + (int)0 * (int)16) * (int)16)] * ((float *)&n448)[(i652 + (i650 + (int)0 * (int)60) * (int)16)];
  i652 += (int)1;
  goto loop_cond_i652;

after_loop_i652:
  i651 += (int)1;
  goto loop_cond_i651;

loop_cond_i655:
  if (i655 > (int)59) goto after_loop_i655; else goto loop_body_i655;

loop_body_i655:
  (void)fprintf (log_file, "index i655 = %d\n", i655);
  (void)fflush (log_file);
  i656 = (int)0;
  goto loop_cond_i656;

after_loop_i655:
  i659 = (int)0;
  goto loop_cond_i659;

loop_cond_i656:
  if (i656 > (int)15) goto after_loop_i656; else goto loop_body_i656;

loop_body_i656:
  (void)fprintf (log_file, "index i656 = %d\n", i656);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n452[i655, i656] := (b2[i656] + n450[i655, i656]);\n");
  (void)fprintf (log_file, "n452[%d]{=%g} = %g = (b2[%d]{=%g} + n450[%d]{=%g})\n", ((double)((float *)&n452)[(i656 + (i655 + (int)0 * (int)60) * (int)16)]), (i656 + (i655 + (int)0 * (int)60) * (int)16), ((double)(b2[(i656 + (int)0 * (int)16)] + ((float *)&n450)[(i656 + (i655 + (int)0 * (int)60) * (int)16)])), (i656 + (int)0 * (int)16), ((double)b2[(i656 + (int)0 * (int)16)]), (i656 + (i655 + (int)0 * (int)60) * (int)16), ((double)((float *)&n450)[(i656 + (i655 + (int)0 * (int)60) * (int)16)]));
  (void)fflush (log_file);
  ((float *)&n452)[(i656 + (i655 + (int)0 * (int)60) * (int)16)] = b2[(i656 + (int)0 * (int)16)] + ((float *)&n450)[(i656 + (i655 + (int)0 * (int)60) * (int)16)];
  i656 += (int)1;
  goto loop_cond_i656;

after_loop_i656:
  i655 += (int)1;
  goto loop_cond_i655;

loop_cond_i659:
  if (i659 > (int)59) goto after_loop_i659; else goto loop_body_i659;

loop_body_i659:
  (void)fprintf (log_file, "index i659 = %d\n", i659);
  (void)fflush (log_file);
  i660 = (int)0;
  goto loop_cond_i660;

after_loop_i659:
  (void)fprintf (log_file, "memset_zero(n456) where before first element = %g\n", ((double)((float *)&n456)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n456), (int)0, (size_t)240);
  i663 = (int)0;
  goto loop_cond_i663;

loop_cond_i660:
  if (i660 > (int)15) goto after_loop_i660; else goto loop_body_i660;

loop_body_i660:
  (void)fprintf (log_file, "index i660 = %d\n", i660);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n454[i659, i660] := relu(n452[i659, i660]);\n");
  (void)fprintf (log_file, "n454[%d]{=%g} = %g = (n452[%d]{=%g} > 0.0 ? n452[%d]{=%g} : 0.0)\n", ((double)((float *)&n454)[(i660 + (i659 + (int)0 * (int)60) * (int)16)]), (i660 + (i659 + (int)0 * (int)60) * (int)16), ((double)((float)(int)((float)0 < ((float *)&n452)[(i660 + (i659 + (int)0 * (int)60) * (int)16)]) * ((float *)&n452)[(i660 + (i659 + (int)0 * (int)60) * (int)16)])), (i660 + (i659 + (int)0 * (int)60) * (int)16), ((double)((float *)&n452)[(i660 + (i659 + (int)0 * (int)60) * (int)16)]), (i660 + (i659 + (int)0 * (int)60) * (int)16), ((double)((float *)&n452)[(i660 + (i659 + (int)0 * (int)60) * (int)16)]));
  (void)fflush (log_file);
  ((float *)&n454)[(i660 + (i659 + (int)0 * (int)60) * (int)16)] = (float)(int)((float)0 < ((float *)&n452)[(i660 + (i659 + (int)0 * (int)60) * (int)16)]) * ((float *)&n452)[(i660 + (i659 + (int)0 * (int)60) * (int)16)];
  i660 += (int)1;
  goto loop_cond_i660;

after_loop_i660:
  i659 += (int)1;
  goto loop_cond_i659;

loop_cond_i663:
  if (i663 > (int)59) goto after_loop_i663; else goto loop_body_i663;

loop_body_i663:
  (void)fprintf (log_file, "index i663 = %d\n", i663);
  (void)fflush (log_file);
  i664 = (int)0;
  goto loop_cond_i664;

after_loop_i663:
  i666 = (int)0;
  goto loop_cond_i666;

loop_cond_i664:
  if (i664 > (int)15) goto after_loop_i664; else goto loop_body_i664;

loop_body_i664:
  (void)fprintf (log_file, "index i664 = %d\n", i664);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n456[i663, 0] := (n456[i663, 0] + (w3[0, i664] * n454[i663, i664]));\n");
  (void)fprintf (log_file, "n456[%d]{=%g} += %g = (w3[%d]{=%g} * n454[%d]{=%g})\n", ((double)((float *)&n456)[((int)0 + (i663 + (int)0 * (int)60) * (int)1)]), ((int)0 + (i663 + (int)0 * (int)60) * (int)1), ((double)(w3[(i664 + ((int)0 + (int)0 * (int)1) * (int)16)] * ((float *)&n454)[(i664 + (i663 + (int)0 * (int)60) * (int)16)])), (i664 + ((int)0 + (int)0 * (int)1) * (int)16), ((double)w3[(i664 + ((int)0 + (int)0 * (int)1) * (int)16)]), (i664 + (i663 + (int)0 * (int)60) * (int)16), ((double)((float *)&n454)[(i664 + (i663 + (int)0 * (int)60) * (int)16)]));
  (void)fflush (log_file);
  ((float *)&n456)[((int)0 + (i663 + (int)0 * (int)60) * (int)1)] += w3[(i664 + ((int)0 + (int)0 * (int)1) * (int)16)] * ((float *)&n454)[(i664 + (i663 + (int)0 * (int)60) * (int)16)];
  i664 += (int)1;
  goto loop_cond_i664;

after_loop_i664:
  i663 += (int)1;
  goto loop_cond_i663;

loop_cond_i666:
  if (i666 > (int)59) goto after_loop_i666; else goto loop_body_i666;

loop_body_i666:
  (void)fprintf (log_file, "index i666 = %d\n", i666);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# mlp[i666, 0] := (b3[0] + n456[i666, 0]);\n");
  (void)fprintf (log_file, "mlp[%d]{=%g} = %g = (b3[%d]{=%g} + n456[%d]{=%g})\n", ((double)((float *)&mlp)[((int)0 + (i666 + (int)0 * (int)60) * (int)1)]), ((int)0 + (i666 + (int)0 * (int)60) * (int)1), ((double)(b3[((int)0 + (int)0 * (int)1)] + ((float *)&n456)[((int)0 + (i666 + (int)0 * (int)60) * (int)1)])), ((int)0 + (int)0 * (int)1), ((double)b3[((int)0 + (int)0 * (int)1)]), ((int)0 + (i666 + (int)0 * (int)60) * (int)1), ((double)((float *)&n456)[((int)0 + (i666 + (int)0 * (int)60) * (int)1)]));
  (void)fflush (log_file);
  ((float *)&mlp)[((int)0 + (i666 + (int)0 * (int)60) * (int)1)] = b3[((int)0 + (int)0 * (int)1)] + ((float *)&n456)[((int)0 + (i666 + (int)0 * (int)60) * (int)1)];
  i666 += (int)1;
  goto loop_cond_i666;

after_loop_i666:
  i668 = (int)0;
  goto loop_cond_i668;

loop_cond_i668:
  if (i668 > (int)59) goto after_loop_i668; else goto loop_body_i668;

loop_body_i668:
  (void)fprintf (log_file, "index i668 = %d\n", i668);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n460[i668, 0] := (expectation[i668, 0] * mlp[i668, 0]);\n");
  (void)fprintf (log_file, "n460[%d]{=%g} = %g = (expectation[%d]{=%g} * mlp[%d]{=%g})\n", ((double)((float *)&n460)[((int)0 + (i668 + (int)0 * (int)60) * (int)1)]), ((int)0 + (i668 + (int)0 * (int)60) * (int)1), ((double)(((float *)&expectation)[((int)0 + (i668 + (int)0 * (int)60) * (int)1)] * ((float *)&mlp)[((int)0 + (i668 + (int)0 * (int)60) * (int)1)])), ((int)0 + (i668 + (int)0 * (int)60) * (int)1), ((double)((float *)&expectation)[((int)0 + (i668 + (int)0 * (int)60) * (int)1)]), ((int)0 + (i668 + (int)0 * (int)60) * (int)1), ((double)((float *)&mlp)[((int)0 + (i668 + (int)0 * (int)60) * (int)1)]));
  (void)fflush (log_file);
  ((float *)&n460)[((int)0 + (i668 + (int)0 * (int)60) * (int)1)] = ((float *)&expectation)[((int)0 + (i668 + (int)0 * (int)60) * (int)1)] * ((float *)&mlp)[((int)0 + (i668 + (int)0 * (int)60) * (int)1)];
  i668 += (int)1;
  goto loop_cond_i668;

after_loop_i668:
  i670 = (int)0;
  goto loop_cond_i670;

loop_cond_i670:
  if (i670 > (int)59) goto after_loop_i670; else goto loop_body_i670;

loop_body_i670:
  (void)fprintf (log_file, "index i670 = %d\n", i670);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n463[i670, 0] := (1.000000 - n460[i670, 0]);\n");
  (void)fprintf (log_file, "n463[%d]{=%g} = %g = (1. - n460[%d]{=%g})\n", ((double)((float *)&n463)[((int)0 + (i670 + (int)0 * (int)60) * (int)1)]), ((int)0 + (i670 + (int)0 * (int)60) * (int)1), ((double)((float)1.000000 - ((float *)&n460)[((int)0 + (i670 + (int)0 * (int)60) * (int)1)])), ((int)0 + (i670 + (int)0 * (int)60) * (int)1), ((double)((float *)&n460)[((int)0 + (i670 + (int)0 * (int)60) * (int)1)]));
  (void)fflush (log_file);
  ((float *)&n463)[((int)0 + (i670 + (int)0 * (int)60) * (int)1)] = (float)1.000000 - ((float *)&n460)[((int)0 + (i670 + (int)0 * (int)60) * (int)1)];
  i670 += (int)1;
  goto loop_cond_i670;

after_loop_i670:
  i672 = (int)0;
  goto loop_cond_i672;

loop_cond_i672:
  if (i672 > (int)59) goto after_loop_i672; else goto loop_body_i672;

loop_body_i672:
  (void)fprintf (log_file, "index i672 = %d\n", i672);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n465[i672, 0] := relu(n463[i672, 0]);\n");
  (void)fprintf (log_file, "n465[%d]{=%g} = %g = (n463[%d]{=%g} > 0.0 ? n463[%d]{=%g} : 0.0)\n", ((double)((float *)&n465)[((int)0 + (i672 + (int)0 * (int)60) * (int)1)]), ((int)0 + (i672 + (int)0 * (int)60) * (int)1), ((double)((float)(int)((float)0 < ((float *)&n463)[((int)0 + (i672 + (int)0 * (int)60) * (int)1)]) * ((float *)&n463)[((int)0 + (i672 + (int)0 * (int)60) * (int)1)])), ((int)0 + (i672 + (int)0 * (int)60) * (int)1), ((double)((float *)&n463)[((int)0 + (i672 + (int)0 * (int)60) * (int)1)]), ((int)0 + (i672 + (int)0 * (int)60) * (int)1), ((double)((float *)&n463)[((int)0 + (i672 + (int)0 * (int)60) * (int)1)]));
  (void)fflush (log_file);
  ((float *)&n465)[((int)0 + (i672 + (int)0 * (int)60) * (int)1)] = (float)(int)((float)0 < ((float *)&n463)[((int)0 + (i672 + (int)0 * (int)60) * (int)1)]) * ((float *)&n463)[((int)0 + (i672 + (int)0 * (int)60) * (int)1)];
  i672 += (int)1;
  goto loop_cond_i672;

after_loop_i672:
  (void)fprintf (log_file, "memset_zero(n468) where before first element = %g\n", ((double)((float *)&n468)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n468), (int)0, (size_t)4);
  i675 = (int)0;
  goto loop_cond_i675;

loop_cond_i675:
  if (i675 > (int)59) goto after_loop_i675; else goto loop_body_i675;

loop_body_i675:
  (void)fprintf (log_file, "index i675 = %d\n", i675);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n468[0] := (n468[0] + n465[i675, 0]);\n");
  (void)fprintf (log_file, "n468[%d]{=%g} += %g = n465[%d]{=%g}\n", ((double)((float *)&n468)[((int)0 + (int)0 * (int)1)]), ((int)0 + (int)0 * (int)1), ((double)((float *)&n465)[((int)0 + (i675 + (int)0 * (int)60) * (int)1)]), ((int)0 + (i675 + (int)0 * (int)60) * (int)1), ((double)((float *)&n465)[((int)0 + (i675 + (int)0 * (int)60) * (int)1)]));
  (void)fflush (log_file);
  ((float *)&n468)[((int)0 + (int)0 * (int)1)] += ((float *)&n465)[((int)0 + (i675 + (int)0 * (int)60) * (int)1)];
  i675 += (int)1;
  goto loop_cond_i675;

after_loop_i675:
  (void)fprintf (log_file, "# scalar_loss[0] := (n468[0] / 120.000000);\n");
  (void)fprintf (log_file, "scalar_loss[%d]{=%g} = %g = (n468[%d]{=%g} / 120.)\n", ((double)scalar_loss[((int)0 + (int)0 * (int)1)]), ((int)0 + (int)0 * (int)1), ((double)(((float *)&n468)[((int)0 + (int)0 * (int)1)] / (float)120.000000)), ((int)0 + (int)0 * (int)1), ((double)((float *)&n468)[((int)0 + (int)0 * (int)1)]));
  (void)fflush (log_file);
  scalar_loss[((int)0 + (int)0 * (int)1)] = ((float *)&n468)[((int)0 + (int)0 * (int)1)] / (float)120.000000;
  (void)fprintf (log_file, "\nCOMMENT: end\n");
  (void)fflush (log_file);
  (void)fprintf (log_file, "\nCOMMENT: scalar_loss zero grads\n");
  (void)fflush (log_file);
  (void)fprintf (log_file, "memset_zero(b3_grad) where before first element = %g\n", ((double)b3_grad[(int)0]));
  (void)fflush (log_file);
  (void)memset (b3_grad, (int)0, (size_t)4);
  (void)fprintf (log_file, "memset_zero(w3_grad) where before first element = %g\n", ((double)w3_grad[(int)0]));
  (void)fflush (log_file);
  (void)memset (w3_grad, (int)0, (size_t)64);
  (void)fprintf (log_file, "memset_zero(b2_grad) where before first element = %g\n", ((double)b2_grad[(int)0]));
  (void)fflush (log_file);
  (void)memset (b2_grad, (int)0, (size_t)64);
  (void)fprintf (log_file, "memset_zero(w2_grad) where before first element = %g\n", ((double)w2_grad[(int)0]));
  (void)fflush (log_file);
  (void)memset (w2_grad, (int)0, (size_t)1024);
  (void)fprintf (log_file, "memset_zero(b1_grad) where before first element = %g\n", ((double)b1_grad[(int)0]));
  (void)fflush (log_file);
  (void)memset (b1_grad, (int)0, (size_t)64);
  (void)fprintf (log_file, "memset_zero(w1_grad) where before first element = %g\n", ((double)w1_grad[(int)0]));
  (void)fflush (log_file);
  (void)memset (w1_grad, (int)0, (size_t)128);
  (void)fprintf (log_file, "memset_zero(n444_grad) where before first element = %g\n", ((double)((float *)&n444_grad)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n444_grad), (int)0, (size_t)3840);
  (void)fprintf (log_file, "memset_zero(n446_grad) where before first element = %g\n", ((double)((float *)&n446_grad)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n446_grad), (int)0, (size_t)3840);
  (void)fprintf (log_file, "memset_zero(n448_grad) where before first element = %g\n", ((double)((float *)&n448_grad)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n448_grad), (int)0, (size_t)3840);
  (void)fprintf (log_file, "memset_zero(n450_grad) where before first element = %g\n", ((double)((float *)&n450_grad)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n450_grad), (int)0, (size_t)3840);
  (void)fprintf (log_file, "memset_zero(n452_grad) where before first element = %g\n", ((double)((float *)&n452_grad)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n452_grad), (int)0, (size_t)3840);
  (void)fprintf (log_file, "memset_zero(n454_grad) where before first element = %g\n", ((double)((float *)&n454_grad)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n454_grad), (int)0, (size_t)3840);
  (void)fprintf (log_file, "memset_zero(n456_grad) where before first element = %g\n", ((double)((float *)&n456_grad)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n456_grad), (int)0, (size_t)240);
  (void)fprintf (log_file, "memset_zero(mlp_grad) where before first element = %g\n", ((double)((float *)&mlp_grad)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&mlp_grad), (int)0, (size_t)240);
  (void)fprintf (log_file, "memset_zero(n460_grad) where before first element = %g\n", ((double)((float *)&n460_grad)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n460_grad), (int)0, (size_t)240);
  (void)fprintf (log_file, "memset_zero(n463_grad) where before first element = %g\n", ((double)((float *)&n463_grad)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n463_grad), (int)0, (size_t)240);
  (void)fprintf (log_file, "memset_zero(n465_grad) where before first element = %g\n", ((double)((float *)&n465_grad)[(int)0]));
  (void)fflush (log_file);
  (void)memset (((float *)&n465_grad), (int)0, (size_t)240);
  (void)fprintf (log_file, "\nCOMMENT: end\n");
  (void)fflush (log_file);
  (void)fprintf (log_file, "\nCOMMENT: scalar_loss bprop\n");
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n475[0] := (-1.000000 * n468[0]);\n");
  (void)fprintf (log_file, "n475[%d]{=%g} = %g = (-1. * n468[%d]{=%g})\n", ((double)((float *)&n475)[((int)0 + (int)0 * (int)1)]), ((int)0 + (int)0 * (int)1), ((double)((float)-1.000000 * ((float *)&n468)[((int)0 + (int)0 * (int)1)])), ((int)0 + (int)0 * (int)1), ((double)((float *)&n468)[((int)0 + (int)0 * (int)1)]));
  (void)fflush (log_file);
  ((float *)&n475)[((int)0 + (int)0 * (int)1)] = (float)-1.000000 * ((float *)&n468)[((int)0 + (int)0 * (int)1)];
  i676 = (int)0;
  goto loop_cond_i676;

loop_cond_i676:
  if (i676 > (int)59) goto after_loop_i676; else goto loop_body_i676;

loop_body_i676:
  (void)fprintf (log_file, "index i676 = %d\n", i676);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n465.grad[i676, 0] := (n465.grad[i676, 0] + 0.008333);\n");
  (void)fprintf (log_file, "n465_grad[%d]{=%g} += %g = 0.0083333333333333332\n", ((double)((float *)&n465_grad)[((int)0 + (i676 + (int)0 * (int)60) * (int)1)]), ((int)0 + (i676 + (int)0 * (int)60) * (int)1), ((double)(float)0.008333));
  (void)fflush (log_file);
  ((float *)&n465_grad)[((int)0 + (i676 + (int)0 * (int)60) * (int)1)] += (float)0.008333;
  i676 += (int)1;
  goto loop_cond_i676;

after_loop_i676:
  i677 = (int)0;
  goto loop_cond_i677;

loop_cond_i677:
  if (i677 > (int)59) goto after_loop_i677; else goto loop_body_i677;

loop_body_i677:
  (void)fprintf (log_file, "index i677 = %d\n", i677);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n463.grad[i677, 0] :=$  (n463.grad[i677, 0] + (n465[i677, 0] > 0.0 ? n465.grad[i677, 0] : 0.0));\n");
  (void)fprintf (log_file, "n463_grad[%d]{=%g} += %g = (n465[%d]{=%g} > 0.0 ? n465_grad[%d]{=%g} : 0.0)\n", ((double)((float *)&n463_grad)[((int)0 + (i677 + (int)0 * (int)60) * (int)1)]), ((int)0 + (i677 + (int)0 * (int)60) * (int)1), ((double)((float)(int)((float)0 < ((float *)&n465)[((int)0 + (i677 + (int)0 * (int)60) * (int)1)]) * ((float *)&n465_grad)[((int)0 + (i677 + (int)0 * (int)60) * (int)1)])), ((int)0 + (i677 + (int)0 * (int)60) * (int)1), ((double)((float *)&n465)[((int)0 + (i677 + (int)0 * (int)60) * (int)1)]), ((int)0 + (i677 + (int)0 * (int)60) * (int)1), ((double)((float *)&n465_grad)[((int)0 + (i677 + (int)0 * (int)60) * (int)1)]));
  (void)fflush (log_file);
  ((float *)&n463_grad)[((int)0 + (i677 + (int)0 * (int)60) * (int)1)] += (float)(int)((float)0 < ((float *)&n465)[((int)0 + (i677 + (int)0 * (int)60) * (int)1)]) * ((float *)&n465_grad)[((int)0 + (i677 + (int)0 * (int)60) * (int)1)];
  i677 += (int)1;
  goto loop_cond_i677;

after_loop_i677:
  i678 = (int)0;
  goto loop_cond_i678;

loop_cond_i678:
  if (i678 > (int)59) goto after_loop_i678; else goto loop_body_i678;

loop_body_i678:
  (void)fprintf (log_file, "index i678 = %d\n", i678);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n460.grad[i678, 0] := (n460.grad[i678, 0] - n463.grad[i678, 0]);\n");
  (void)fprintf (log_file, "n460_grad[%d]{=%g} -= %g = n463_grad[%d]{=%g}\n", ((double)((float *)&n460_grad)[((int)0 + (i678 + (int)0 * (int)60) * (int)1)]), ((int)0 + (i678 + (int)0 * (int)60) * (int)1), ((double)((float *)&n463_grad)[((int)0 + (i678 + (int)0 * (int)60) * (int)1)]), ((int)0 + (i678 + (int)0 * (int)60) * (int)1), ((double)((float *)&n463_grad)[((int)0 + (i678 + (int)0 * (int)60) * (int)1)]));
  (void)fflush (log_file);
  ((float *)&n460_grad)[((int)0 + (i678 + (int)0 * (int)60) * (int)1)] -= ((float *)&n463_grad)[((int)0 + (i678 + (int)0 * (int)60) * (int)1)];
  i678 += (int)1;
  goto loop_cond_i678;

after_loop_i678:
  i679 = (int)0;
  goto loop_cond_i679;

loop_cond_i679:
  if (i679 > (int)59) goto after_loop_i679; else goto loop_body_i679;

loop_body_i679:
  (void)fprintf (log_file, "index i679 = %d\n", i679);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# mlp.grad[i679, 0] :=$  (mlp.grad[i679, 0] + (expectation[i679, 0] * n460.grad[i679, 0]));\n");
  (void)fprintf (log_file, "mlp_grad[%d]{=%g} += %g = (expectation[%d]{=%g} * n460_grad[%d]{=%g})\n", ((double)((float *)&mlp_grad)[((int)0 + (i679 + (int)0 * (int)60) * (int)1)]), ((int)0 + (i679 + (int)0 * (int)60) * (int)1), ((double)(((float *)&expectation)[((int)0 + (i679 + (int)0 * (int)60) * (int)1)] * ((float *)&n460_grad)[((int)0 + (i679 + (int)0 * (int)60) * (int)1)])), ((int)0 + (i679 + (int)0 * (int)60) * (int)1), ((double)((float *)&expectation)[((int)0 + (i679 + (int)0 * (int)60) * (int)1)]), ((int)0 + (i679 + (int)0 * (int)60) * (int)1), ((double)((float *)&n460_grad)[((int)0 + (i679 + (int)0 * (int)60) * (int)1)]));
  (void)fflush (log_file);
  ((float *)&mlp_grad)[((int)0 + (i679 + (int)0 * (int)60) * (int)1)] += ((float *)&expectation)[((int)0 + (i679 + (int)0 * (int)60) * (int)1)] * ((float *)&n460_grad)[((int)0 + (i679 + (int)0 * (int)60) * (int)1)];
  i679 += (int)1;
  goto loop_cond_i679;

after_loop_i679:
  i680 = (int)0;
  goto loop_cond_i680;

loop_cond_i680:
  if (i680 > (int)59) goto after_loop_i680; else goto loop_body_i680;

loop_body_i680:
  (void)fprintf (log_file, "index i680 = %d\n", i680);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# b3.grad[0] := (b3.grad[0] + mlp.grad[i680, 0]);\n");
  (void)fprintf (log_file, "b3_grad[%d]{=%g} += %g = mlp_grad[%d]{=%g}\n", ((double)b3_grad[((int)0 + (int)0 * (int)1)]), ((int)0 + (int)0 * (int)1), ((double)((float *)&mlp_grad)[((int)0 + (i680 + (int)0 * (int)60) * (int)1)]), ((int)0 + (i680 + (int)0 * (int)60) * (int)1), ((double)((float *)&mlp_grad)[((int)0 + (i680 + (int)0 * (int)60) * (int)1)]));
  (void)fflush (log_file);
  b3_grad[((int)0 + (int)0 * (int)1)] += ((float *)&mlp_grad)[((int)0 + (i680 + (int)0 * (int)60) * (int)1)];
  i680 += (int)1;
  goto loop_cond_i680;

after_loop_i680:
  i681 = (int)0;
  goto loop_cond_i681;

loop_cond_i681:
  if (i681 > (int)59) goto after_loop_i681; else goto loop_body_i681;

loop_body_i681:
  (void)fprintf (log_file, "index i681 = %d\n", i681);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n456.grad[i681, 0] := (n456.grad[i681, 0] + mlp.grad[i681, 0]);\n");
  (void)fprintf (log_file, "n456_grad[%d]{=%g} += %g = mlp_grad[%d]{=%g}\n", ((double)((float *)&n456_grad)[((int)0 + (i681 + (int)0 * (int)60) * (int)1)]), ((int)0 + (i681 + (int)0 * (int)60) * (int)1), ((double)((float *)&mlp_grad)[((int)0 + (i681 + (int)0 * (int)60) * (int)1)]), ((int)0 + (i681 + (int)0 * (int)60) * (int)1), ((double)((float *)&mlp_grad)[((int)0 + (i681 + (int)0 * (int)60) * (int)1)]));
  (void)fflush (log_file);
  ((float *)&n456_grad)[((int)0 + (i681 + (int)0 * (int)60) * (int)1)] += ((float *)&mlp_grad)[((int)0 + (i681 + (int)0 * (int)60) * (int)1)];
  i681 += (int)1;
  goto loop_cond_i681;

after_loop_i681:
  i682 = (int)0;
  goto loop_cond_i682;

loop_cond_i682:
  if (i682 > (int)59) goto after_loop_i682; else goto loop_body_i682;

loop_body_i682:
  (void)fprintf (log_file, "index i682 = %d\n", i682);
  (void)fflush (log_file);
  i683 = (int)0;
  goto loop_cond_i683;

after_loop_i682:
  i684 = (int)0;
  goto loop_cond_i684;

loop_cond_i683:
  if (i683 > (int)15) goto after_loop_i683; else goto loop_body_i683;

loop_body_i683:
  (void)fprintf (log_file, "index i683 = %d\n", i683);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# w3.grad[0, i683] :=$  (w3.grad[0, i683] + (n456.grad[i682, 0] * n454[i682, i683]));\n");
  (void)fprintf (log_file, "w3_grad[%d]{=%g} += %g = (n456_grad[%d]{=%g} * n454[%d]{=%g})\n", ((double)w3_grad[(i683 + ((int)0 + (int)0 * (int)1) * (int)16)]), (i683 + ((int)0 + (int)0 * (int)1) * (int)16), ((double)(((float *)&n456_grad)[((int)0 + (i682 + (int)0 * (int)60) * (int)1)] * ((float *)&n454)[(i683 + (i682 + (int)0 * (int)60) * (int)16)])), ((int)0 + (i682 + (int)0 * (int)60) * (int)1), ((double)((float *)&n456_grad)[((int)0 + (i682 + (int)0 * (int)60) * (int)1)]), (i683 + (i682 + (int)0 * (int)60) * (int)16), ((double)((float *)&n454)[(i683 + (i682 + (int)0 * (int)60) * (int)16)]));
  (void)fflush (log_file);
  w3_grad[(i683 + ((int)0 + (int)0 * (int)1) * (int)16)] += ((float *)&n456_grad)[((int)0 + (i682 + (int)0 * (int)60) * (int)1)] * ((float *)&n454)[(i683 + (i682 + (int)0 * (int)60) * (int)16)];
  i683 += (int)1;
  goto loop_cond_i683;

after_loop_i683:
  i682 += (int)1;
  goto loop_cond_i682;

loop_cond_i684:
  if (i684 > (int)59) goto after_loop_i684; else goto loop_body_i684;

loop_body_i684:
  (void)fprintf (log_file, "index i684 = %d\n", i684);
  (void)fflush (log_file);
  i685 = (int)0;
  goto loop_cond_i685;

after_loop_i684:
  i686 = (int)0;
  goto loop_cond_i686;

loop_cond_i685:
  if (i685 > (int)15) goto after_loop_i685; else goto loop_body_i685;

loop_body_i685:
  (void)fprintf (log_file, "index i685 = %d\n", i685);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n454.grad[i684, i685] :=$  (n454.grad[i684, i685] + (w3[0, i685] * n456.grad[i684, 0]));\n");
  (void)fprintf (log_file, "n454_grad[%d]{=%g} += %g = (w3[%d]{=%g} * n456_grad[%d]{=%g})\n", ((double)((float *)&n454_grad)[(i685 + (i684 + (int)0 * (int)60) * (int)16)]), (i685 + (i684 + (int)0 * (int)60) * (int)16), ((double)(w3[(i685 + ((int)0 + (int)0 * (int)1) * (int)16)] * ((float *)&n456_grad)[((int)0 + (i684 + (int)0 * (int)60) * (int)1)])), (i685 + ((int)0 + (int)0 * (int)1) * (int)16), ((double)w3[(i685 + ((int)0 + (int)0 * (int)1) * (int)16)]), ((int)0 + (i684 + (int)0 * (int)60) * (int)1), ((double)((float *)&n456_grad)[((int)0 + (i684 + (int)0 * (int)60) * (int)1)]));
  (void)fflush (log_file);
  ((float *)&n454_grad)[(i685 + (i684 + (int)0 * (int)60) * (int)16)] += w3[(i685 + ((int)0 + (int)0 * (int)1) * (int)16)] * ((float *)&n456_grad)[((int)0 + (i684 + (int)0 * (int)60) * (int)1)];
  i685 += (int)1;
  goto loop_cond_i685;

after_loop_i685:
  i684 += (int)1;
  goto loop_cond_i684;

loop_cond_i686:
  if (i686 > (int)59) goto after_loop_i686; else goto loop_body_i686;

loop_body_i686:
  (void)fprintf (log_file, "index i686 = %d\n", i686);
  (void)fflush (log_file);
  i687 = (int)0;
  goto loop_cond_i687;

after_loop_i686:
  i688 = (int)0;
  goto loop_cond_i688;

loop_cond_i687:
  if (i687 > (int)15) goto after_loop_i687; else goto loop_body_i687;

loop_body_i687:
  (void)fprintf (log_file, "index i687 = %d\n", i687);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n452.grad[i686, i687] :=$  (n452.grad[i686, i687] +$   (n454[i686, i687] > 0.0 ? n454.grad[i686, i687] : 0.0));\n");
  (void)fprintf (log_file, "n452_grad[%d]{=%g} += %g = (n454[%d]{=%g} > 0.0 ? n454_grad[%d]{=%g} : 0.0)\n", ((double)((float *)&n452_grad)[(i687 + (i686 + (int)0 * (int)60) * (int)16)]), (i687 + (i686 + (int)0 * (int)60) * (int)16), ((double)((float)(int)((float)0 < ((float *)&n454)[(i687 + (i686 + (int)0 * (int)60) * (int)16)]) * ((float *)&n454_grad)[(i687 + (i686 + (int)0 * (int)60) * (int)16)])), (i687 + (i686 + (int)0 * (int)60) * (int)16), ((double)((float *)&n454)[(i687 + (i686 + (int)0 * (int)60) * (int)16)]), (i687 + (i686 + (int)0 * (int)60) * (int)16), ((double)((float *)&n454_grad)[(i687 + (i686 + (int)0 * (int)60) * (int)16)]));
  (void)fflush (log_file);
  ((float *)&n452_grad)[(i687 + (i686 + (int)0 * (int)60) * (int)16)] += (float)(int)((float)0 < ((float *)&n454)[(i687 + (i686 + (int)0 * (int)60) * (int)16)]) * ((float *)&n454_grad)[(i687 + (i686 + (int)0 * (int)60) * (int)16)];
  i687 += (int)1;
  goto loop_cond_i687;

after_loop_i687:
  i686 += (int)1;
  goto loop_cond_i686;

loop_cond_i688:
  if (i688 > (int)59) goto after_loop_i688; else goto loop_body_i688;

loop_body_i688:
  (void)fprintf (log_file, "index i688 = %d\n", i688);
  (void)fflush (log_file);
  i689 = (int)0;
  goto loop_cond_i689;

after_loop_i688:
  i690 = (int)0;
  goto loop_cond_i690;

loop_cond_i689:
  if (i689 > (int)15) goto after_loop_i689; else goto loop_body_i689;

loop_body_i689:
  (void)fprintf (log_file, "index i689 = %d\n", i689);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# b2.grad[i689] := (b2.grad[i689] + n452.grad[i688, i689]);\n");
  (void)fprintf (log_file, "b2_grad[%d]{=%g} += %g = n452_grad[%d]{=%g}\n", ((double)b2_grad[(i689 + (int)0 * (int)16)]), (i689 + (int)0 * (int)16), ((double)((float *)&n452_grad)[(i689 + (i688 + (int)0 * (int)60) * (int)16)]), (i689 + (i688 + (int)0 * (int)60) * (int)16), ((double)((float *)&n452_grad)[(i689 + (i688 + (int)0 * (int)60) * (int)16)]));
  (void)fflush (log_file);
  b2_grad[(i689 + (int)0 * (int)16)] += ((float *)&n452_grad)[(i689 + (i688 + (int)0 * (int)60) * (int)16)];
  i689 += (int)1;
  goto loop_cond_i689;

after_loop_i689:
  i688 += (int)1;
  goto loop_cond_i688;

loop_cond_i690:
  if (i690 > (int)59) goto after_loop_i690; else goto loop_body_i690;

loop_body_i690:
  (void)fprintf (log_file, "index i690 = %d\n", i690);
  (void)fflush (log_file);
  i691 = (int)0;
  goto loop_cond_i691;

after_loop_i690:
  i692 = (int)0;
  goto loop_cond_i692;

loop_cond_i691:
  if (i691 > (int)15) goto after_loop_i691; else goto loop_body_i691;

loop_body_i691:
  (void)fprintf (log_file, "index i691 = %d\n", i691);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n450.grad[i690, i691] := (n450.grad[i690, i691] + n452.grad[i690, i691]);\n");
  (void)fprintf (log_file, "n450_grad[%d]{=%g} += %g = n452_grad[%d]{=%g}\n", ((double)((float *)&n450_grad)[(i691 + (i690 + (int)0 * (int)60) * (int)16)]), (i691 + (i690 + (int)0 * (int)60) * (int)16), ((double)((float *)&n452_grad)[(i691 + (i690 + (int)0 * (int)60) * (int)16)]), (i691 + (i690 + (int)0 * (int)60) * (int)16), ((double)((float *)&n452_grad)[(i691 + (i690 + (int)0 * (int)60) * (int)16)]));
  (void)fflush (log_file);
  ((float *)&n450_grad)[(i691 + (i690 + (int)0 * (int)60) * (int)16)] += ((float *)&n452_grad)[(i691 + (i690 + (int)0 * (int)60) * (int)16)];
  i691 += (int)1;
  goto loop_cond_i691;

after_loop_i691:
  i690 += (int)1;
  goto loop_cond_i690;

loop_cond_i692:
  if (i692 > (int)59) goto after_loop_i692; else goto loop_body_i692;

loop_body_i692:
  (void)fprintf (log_file, "index i692 = %d\n", i692);
  (void)fflush (log_file);
  i693 = (int)0;
  goto loop_cond_i693;

after_loop_i692:
  i695 = (int)0;
  goto loop_cond_i695;

loop_cond_i693:
  if (i693 > (int)15) goto after_loop_i693; else goto loop_body_i693;

loop_body_i693:
  (void)fprintf (log_file, "index i693 = %d\n", i693);
  (void)fflush (log_file);
  i694 = (int)0;
  goto loop_cond_i694;

after_loop_i693:
  i692 += (int)1;
  goto loop_cond_i692;

loop_cond_i694:
  if (i694 > (int)15) goto after_loop_i694; else goto loop_body_i694;

loop_body_i694:
  (void)fprintf (log_file, "index i694 = %d\n", i694);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# w2.grad[i693, i694] :=$  (w2.grad[i693, i694] + (n450.grad[i692, i693] * n448[i692, i694]));\n");
  (void)fprintf (log_file, "w2_grad[%d]{=%g} += %g = (n450_grad[%d]{=%g} * n448[%d]{=%g})\n", ((double)w2_grad[(i694 + (i693 + (int)0 * (int)16) * (int)16)]), (i694 + (i693 + (int)0 * (int)16) * (int)16), ((double)(((float *)&n450_grad)[(i693 + (i692 + (int)0 * (int)60) * (int)16)] * ((float *)&n448)[(i694 + (i692 + (int)0 * (int)60) * (int)16)])), (i693 + (i692 + (int)0 * (int)60) * (int)16), ((double)((float *)&n450_grad)[(i693 + (i692 + (int)0 * (int)60) * (int)16)]), (i694 + (i692 + (int)0 * (int)60) * (int)16), ((double)((float *)&n448)[(i694 + (i692 + (int)0 * (int)60) * (int)16)]));
  (void)fflush (log_file);
  w2_grad[(i694 + (i693 + (int)0 * (int)16) * (int)16)] += ((float *)&n450_grad)[(i693 + (i692 + (int)0 * (int)60) * (int)16)] * ((float *)&n448)[(i694 + (i692 + (int)0 * (int)60) * (int)16)];
  i694 += (int)1;
  goto loop_cond_i694;

after_loop_i694:
  i693 += (int)1;
  goto loop_cond_i693;

loop_cond_i695:
  if (i695 > (int)59) goto after_loop_i695; else goto loop_body_i695;

loop_body_i695:
  (void)fprintf (log_file, "index i695 = %d\n", i695);
  (void)fflush (log_file);
  i696 = (int)0;
  goto loop_cond_i696;

after_loop_i695:
  i698 = (int)0;
  goto loop_cond_i698;

loop_cond_i696:
  if (i696 > (int)15) goto after_loop_i696; else goto loop_body_i696;

loop_body_i696:
  (void)fprintf (log_file, "index i696 = %d\n", i696);
  (void)fflush (log_file);
  i697 = (int)0;
  goto loop_cond_i697;

after_loop_i696:
  i695 += (int)1;
  goto loop_cond_i695;

loop_cond_i697:
  if (i697 > (int)15) goto after_loop_i697; else goto loop_body_i697;

loop_body_i697:
  (void)fprintf (log_file, "index i697 = %d\n", i697);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n448.grad[i695, i697] :=$  (n448.grad[i695, i697] + (w2[i696, i697] * n450.grad[i695, i696]));\n");
  (void)fprintf (log_file, "n448_grad[%d]{=%g} += %g = (w2[%d]{=%g} * n450_grad[%d]{=%g})\n", ((double)((float *)&n448_grad)[(i697 + (i695 + (int)0 * (int)60) * (int)16)]), (i697 + (i695 + (int)0 * (int)60) * (int)16), ((double)(w2[(i697 + (i696 + (int)0 * (int)16) * (int)16)] * ((float *)&n450_grad)[(i696 + (i695 + (int)0 * (int)60) * (int)16)])), (i697 + (i696 + (int)0 * (int)16) * (int)16), ((double)w2[(i697 + (i696 + (int)0 * (int)16) * (int)16)]), (i696 + (i695 + (int)0 * (int)60) * (int)16), ((double)((float *)&n450_grad)[(i696 + (i695 + (int)0 * (int)60) * (int)16)]));
  (void)fflush (log_file);
  ((float *)&n448_grad)[(i697 + (i695 + (int)0 * (int)60) * (int)16)] += w2[(i697 + (i696 + (int)0 * (int)16) * (int)16)] * ((float *)&n450_grad)[(i696 + (i695 + (int)0 * (int)60) * (int)16)];
  i697 += (int)1;
  goto loop_cond_i697;

after_loop_i697:
  i696 += (int)1;
  goto loop_cond_i696;

loop_cond_i698:
  if (i698 > (int)59) goto after_loop_i698; else goto loop_body_i698;

loop_body_i698:
  (void)fprintf (log_file, "index i698 = %d\n", i698);
  (void)fflush (log_file);
  i699 = (int)0;
  goto loop_cond_i699;

after_loop_i698:
  i700 = (int)0;
  goto loop_cond_i700;

loop_cond_i699:
  if (i699 > (int)15) goto after_loop_i699; else goto loop_body_i699;

loop_body_i699:
  (void)fprintf (log_file, "index i699 = %d\n", i699);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n446.grad[i698, i699] :=$  (n446.grad[i698, i699] +$   (n448[i698, i699] > 0.0 ? n448.grad[i698, i699] : 0.0));\n");
  (void)fprintf (log_file, "n446_grad[%d]{=%g} += %g = (n448[%d]{=%g} > 0.0 ? n448_grad[%d]{=%g} : 0.0)\n", ((double)((float *)&n446_grad)[(i699 + (i698 + (int)0 * (int)60) * (int)16)]), (i699 + (i698 + (int)0 * (int)60) * (int)16), ((double)((float)(int)((float)0 < ((float *)&n448)[(i699 + (i698 + (int)0 * (int)60) * (int)16)]) * ((float *)&n448_grad)[(i699 + (i698 + (int)0 * (int)60) * (int)16)])), (i699 + (i698 + (int)0 * (int)60) * (int)16), ((double)((float *)&n448)[(i699 + (i698 + (int)0 * (int)60) * (int)16)]), (i699 + (i698 + (int)0 * (int)60) * (int)16), ((double)((float *)&n448_grad)[(i699 + (i698 + (int)0 * (int)60) * (int)16)]));
  (void)fflush (log_file);
  ((float *)&n446_grad)[(i699 + (i698 + (int)0 * (int)60) * (int)16)] += (float)(int)((float)0 < ((float *)&n448)[(i699 + (i698 + (int)0 * (int)60) * (int)16)]) * ((float *)&n448_grad)[(i699 + (i698 + (int)0 * (int)60) * (int)16)];
  i699 += (int)1;
  goto loop_cond_i699;

after_loop_i699:
  i698 += (int)1;
  goto loop_cond_i698;

loop_cond_i700:
  if (i700 > (int)59) goto after_loop_i700; else goto loop_body_i700;

loop_body_i700:
  (void)fprintf (log_file, "index i700 = %d\n", i700);
  (void)fflush (log_file);
  i701 = (int)0;
  goto loop_cond_i701;

after_loop_i700:
  i702 = (int)0;
  goto loop_cond_i702;

loop_cond_i701:
  if (i701 > (int)15) goto after_loop_i701; else goto loop_body_i701;

loop_body_i701:
  (void)fprintf (log_file, "index i701 = %d\n", i701);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# b1.grad[i701] := (b1.grad[i701] + n446.grad[i700, i701]);\n");
  (void)fprintf (log_file, "b1_grad[%d]{=%g} += %g = n446_grad[%d]{=%g}\n", ((double)b1_grad[(i701 + (int)0 * (int)16)]), (i701 + (int)0 * (int)16), ((double)((float *)&n446_grad)[(i701 + (i700 + (int)0 * (int)60) * (int)16)]), (i701 + (i700 + (int)0 * (int)60) * (int)16), ((double)((float *)&n446_grad)[(i701 + (i700 + (int)0 * (int)60) * (int)16)]));
  (void)fflush (log_file);
  b1_grad[(i701 + (int)0 * (int)16)] += ((float *)&n446_grad)[(i701 + (i700 + (int)0 * (int)60) * (int)16)];
  i701 += (int)1;
  goto loop_cond_i701;

after_loop_i701:
  i700 += (int)1;
  goto loop_cond_i700;

loop_cond_i702:
  if (i702 > (int)59) goto after_loop_i702; else goto loop_body_i702;

loop_body_i702:
  (void)fprintf (log_file, "index i702 = %d\n", i702);
  (void)fflush (log_file);
  i703 = (int)0;
  goto loop_cond_i703;

after_loop_i702:
  i704 = (int)0;
  goto loop_cond_i704;

loop_cond_i703:
  if (i703 > (int)15) goto after_loop_i703; else goto loop_body_i703;

loop_body_i703:
  (void)fprintf (log_file, "index i703 = %d\n", i703);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# n444.grad[i702, i703] := (n444.grad[i702, i703] + n446.grad[i702, i703]);\n");
  (void)fprintf (log_file, "n444_grad[%d]{=%g} += %g = n446_grad[%d]{=%g}\n", ((double)((float *)&n444_grad)[(i703 + (i702 + (int)0 * (int)60) * (int)16)]), (i703 + (i702 + (int)0 * (int)60) * (int)16), ((double)((float *)&n446_grad)[(i703 + (i702 + (int)0 * (int)60) * (int)16)]), (i703 + (i702 + (int)0 * (int)60) * (int)16), ((double)((float *)&n446_grad)[(i703 + (i702 + (int)0 * (int)60) * (int)16)]));
  (void)fflush (log_file);
  ((float *)&n444_grad)[(i703 + (i702 + (int)0 * (int)60) * (int)16)] += ((float *)&n446_grad)[(i703 + (i702 + (int)0 * (int)60) * (int)16)];
  i703 += (int)1;
  goto loop_cond_i703;

after_loop_i703:
  i702 += (int)1;
  goto loop_cond_i702;

loop_cond_i704:
  if (i704 > (int)59) goto after_loop_i704; else goto loop_body_i704;

loop_body_i704:
  (void)fprintf (log_file, "index i704 = %d\n", i704);
  (void)fflush (log_file);
  i705 = (int)0;
  goto loop_cond_i705;

after_loop_i704:
  (void)fprintf (log_file, "\nCOMMENT: end\n");
  (void)fflush (log_file);
  (void)fprintf (log_file, "\nCOMMENT: end\n");
  (void)fflush (log_file);
  (void)fclose (log_file);
  return;

loop_cond_i705:
  if (i705 > (int)15) goto after_loop_i705; else goto loop_body_i705;

loop_body_i705:
  (void)fprintf (log_file, "index i705 = %d\n", i705);
  (void)fflush (log_file);
  i706 = (int)0;
  goto loop_cond_i706;

after_loop_i705:
  i704 += (int)1;
  goto loop_cond_i704;

loop_cond_i706:
  if (i706 > (int)1) goto after_loop_i706; else goto loop_body_i706;

loop_body_i706:
  (void)fprintf (log_file, "index i706 = %d\n", i706);
  (void)fflush (log_file);
  (void)fprintf (log_file, "# w1.grad[i705, i706] :=$  (w1.grad[i705, i706] + (n444.grad[i704, i705] * input[i704, i706]));\n");
  (void)fprintf (log_file, "w1_grad[%d]{=%g} += %g = (n444_grad[%d]{=%g} * input[%d]{=%g})\n", ((double)w1_grad[(i706 + (i705 + (int)0 * (int)16) * (int)2)]), (i706 + (i705 + (int)0 * (int)16) * (int)2), ((double)(((float *)&n444_grad)[(i705 + (i704 + (int)0 * (int)60) * (int)16)] * ((float *)&input)[(i706 + (i704 + (int)0 * (int)60) * (int)2)])), (i705 + (i704 + (int)0 * (int)60) * (int)16), ((double)((float *)&n444_grad)[(i705 + (i704 + (int)0 * (int)60) * (int)16)]), (i706 + (i704 + (int)0 * (int)60) * (int)2), ((double)((float *)&input)[(i706 + (i704 + (int)0 * (int)60) * (int)2)]));
  (void)fflush (log_file);
  w1_grad[(i706 + (i705 + (int)0 * (int)16) * (int)2)] += ((float *)&n444_grad)[(i705 + (i704 + (int)0 * (int)60) * (int)16)] * ((float *)&input)[(i706 + (i704 + (int)0 * (int)60) * (int)2)];
  i706 += (int)1;
  goto loop_cond_i706;

after_loop_i706:
  i705 += (int)1;
  goto loop_cond_i705;
}

extern FILE *
fopen (const char * filename, const char * mode); /* (imported) */

extern void
fflush (FILE * f); /* (imported) */

extern void *
fclose (FILE * f); /* (imported) */

