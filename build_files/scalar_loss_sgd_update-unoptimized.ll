
scalar_loss_sgd_update (i157 : [0..19], i158):
  /* scalar_loss sgd update */
  /* b1 param sgd step */
  n169[0] := 0.000200;
  for i240 = 0 to 15 {
    n170[i240] := (n169[0] * b1[i240]);
  }
  for i242 = 0 to 15 {
    sgd_delta_b1[i242] := (b1.grad[i242] + n170[i242]);
  }
  n157[0] := 20.000000;
  n158[0] := i158;
  n159[0] := 20.000000;
  n160[0] := 2.000000;
  n161[0] := (n160[0] * n159[0]);
  n162[0] := (n161[0] - n158[0]);
  n163[0] := 0.100000;
  n164[0] := (n163[0] * n162[0]);
  learning_rate[0] := (n164[0] / n157[0]);
  for i244 = 0 to 15 {
    n168[i244] := (learning_rate[0] * sgd_delta_b1[i244]);
  }
  for i246 = 0 to 15 {
    b1[i246] := (b1[i246] - n168[i246]);
  }
  /* end */
  /* b2 param sgd step */
  n174[0] := 0.000200;
  for i248 = 0 to 15 {
    n175[i248] := (n174[0] * b2[i248]);
  }
  for i250 = 0 to 15 {
    sgd_delta_b2[i250] := (b2.grad[i250] + n175[i250]);
  }
  
  for i252 = 0 to 15 {
    n173[i252] := (learning_rate[0] * sgd_delta_b2[i252]);
  }
  for i254 = 0 to 15 {
    b2[i254] := (b2[i254] - n173[i254]);
  }
  /* end */
  /* b3 param sgd step */
  n179[0] := 0.000200;
  n180[0] := (n179[0] * b3[0]);
  sgd_delta_b3[0] := (b3.grad[0] + n180[0]);
  
  n178[0] := (learning_rate[0] * sgd_delta_b3[0]);
  b3[0] := (b3[0] - n178[0]);
  /* end */
  /* w1 param sgd step */
  n184[0] := 0.000200;
  for i257 = 0 to 15 {
    for i258 = 0 to 1 {
      n185[i257, i258] := (n184[0] * w1[i257, i258]);
    }
  }
  for i261 = 0 to 15 {
    for i262 = 0 to 1 {
      sgd_delta_w1[i261, i262] := (w1.grad[i261, i262] + n185[i261, i262]);
    }
  }
  
  for i265 = 0 to 15 {
    for i266 = 0 to 1 {
      n183[i265, i266] := (learning_rate[0] * sgd_delta_w1[i265, i266]);
    }
  }
  for i269 = 0 to 15 {
    for i270 = 0 to 1 {
      w1[i269, i270] := (w1[i269, i270] - n183[i269, i270]);
    }
  }
  /* end */
  /* w2 param sgd step */
  n189[0] := 0.000200;
  for i273 = 0 to 15 {
    for i274 = 0 to 15 {
      n190[i273, i274] := (n189[0] * w2[i273, i274]);
    }
  }
  for i277 = 0 to 15 {
    for i278 = 0 to 15 {
      sgd_delta_w2[i277, i278] := (w2.grad[i277, i278] + n190[i277, i278]);
    }
  }
  
  for i281 = 0 to 15 {
    for i282 = 0 to 15 {
      n188[i281, i282] := (learning_rate[0] * sgd_delta_w2[i281, i282]);
    }
  }
  for i285 = 0 to 15 {
    for i286 = 0 to 15 {
      w2[i285, i286] := (w2[i285, i286] - n188[i285, i286]);
    }
  }
  /* end */
  /* w3 param sgd step */
  n194[0] := 0.000200;
  for i288 = 0 to 15 {
    n195[0, i288] := (n194[0] * w3[0, i288]);
  }
  for i290 = 0 to 15 {
    sgd_delta_w3[0, i290] := (w3.grad[0, i290] + n195[0, i290]);
  }
  
  for i292 = 0 to 15 {
    n193[0, i292] := (learning_rate[0] * sgd_delta_w3[0, i292]);
  }
  for i294 = 0 to 15 {
    w3[0, i294] := (w3[0, i294] - n193[0, i294]);
  }
  /* end */
  /* end */