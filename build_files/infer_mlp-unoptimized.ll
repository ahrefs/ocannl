
infer_mlp ():
  /* infer mlp */
  
  zero_out n91;
  for i141 = 0 to 15 {
    for i142 = 0 to 1 {
      n91[i141] := (n91[i141] + (w1[i141, i142] * infer[i142]));
    }
  }
  for i144 = 0 to 15 {
    n93[i144] := (b1[i144] + n91[i144]);
  }
  for i146 = 0 to 15 {
    n95[i146] := relu(n93[i146]);
  }
  zero_out n97;
  for i149 = 0 to 15 {
    for i150 = 0 to 15 {
      n97[i149] := (n97[i149] + (w2[i149, i150] * n95[i150]));
    }
  }
  for i152 = 0 to 15 {
    n99[i152] := (b2[i152] + n97[i152]);
  }
  for i154 = 0 to 15 {
    n101[i154] := relu(n99[i154]);
  }
  zero_out n103;
  for i156 = 0 to 15 {
    n103[0] := (n103[0] + (w3[0, i156] * n101[i156]));
  }
  mlp[0] := (b3[0] + n103[0]);
  /* end */