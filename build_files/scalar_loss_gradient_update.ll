
scalar_loss_gradient_update (i157 : [0..19], i158):
  /* scalar_loss gradient update */
  /* scalar_loss fwd */
  for i160 = 0 to 119 {
    expectation[i160, 0] := moons_classes[i157, i160, 0];
  }
  for i163 = 0 to 119 {
    for i164 = 0 to 1 {
      input[i163, i164] := moons_flat[i157, i163, i164];
    }
  }
  zero_out n123;
  for i168 = 0 to 119 {
    for i169 = 0 to 15 {
      for i170 = 0 to 1 {
        n123[i168, i169] := (n123[i168, i169] + (w1[i169, i170] * input[i168, i170]));
      }
    }
  }
  for i173 = 0 to 119 {
    for i174 = 0 to 15 {
      n125[i173, i174] := (b1[i174] + n123[i173, i174]);
    }
  }
  for i177 = 0 to 119 {
    for i178 = 0 to 15 {
      n127[i177, i178] := relu(n125[i177, i178]);
    }
  }
  zero_out n129;
  for i182 = 0 to 119 {
    for i183 = 0 to 15 {
      for i184 = 0 to 15 {
        n129[i182, i183] := (n129[i182, i183] + (w2[i183, i184] * n127[i182, i184]));
      }
    }
  }
  for i187 = 0 to 119 {
    for i188 = 0 to 15 {
      n131[i187, i188] := (b2[i188] + n129[i187, i188]);
    }
  }
  for i191 = 0 to 119 {
    for i192 = 0 to 15 {
      n133[i191, i192] := relu(n131[i191, i192]);
    }
  }
  zero_out n135;
  for i195 = 0 to 119 {
    for i196 = 0 to 15 {
      n135[i195, 0] := (n135[i195, 0] + (w3[0, i196] * n133[i195, i196]));
    }
  }
  for i198 = 0 to 119 {
    mlp[i198, 0] := (b3[0] + n135[i198, 0]);
  }
  for i200 = 0 to 119 {
    n139[i200, 0] := (expectation[i200, 0] * mlp[i200, 0]);
  }
  for i202 = 0 to 119 {
    n142[i202, 0] := (1.000000 - n139[i202, 0]);
  }
  for i204 = 0 to 119 {
    n144[i204, 0] := relu(n142[i204, 0]);
  }
  zero_out n147;
  for i207 = 0 to 119 {
    n147[0] := (n147[0] + n144[i207, 0]);
  }
  scalar_loss[0] := (n147[0] / 120.000000);
  /* end */
  /* scalar_loss zero grads */
  zero_out b3.grad;
  zero_out w3.grad;
  zero_out b2.grad;
  zero_out w2.grad;
  zero_out b1.grad;
  zero_out w1.grad;
  zero_out n123.grad;
  zero_out n125.grad;
  zero_out n127.grad;
  zero_out n129.grad;
  zero_out n131.grad;
  zero_out n133.grad;
  zero_out n135.grad;
  zero_out mlp.grad;
  zero_out n139.grad;
  zero_out n142.grad;
  zero_out n144.grad;
  /* end */
  /* scalar_loss bprop */
  n154[0] := (-1.000000 * n147[0]);
  for i208 = 0 to 119 {
    n144.grad[i208, 0] := (n144.grad[i208, 0] + 0.008333);
  }
  for i209 = 0 to 119 {
    n142.grad[i209, 0] := (n142.grad[i209, 0] + (n144[i209, 0] > 0.0 ? n144.grad[i209, 0] : 0.0));
  }
  for i210 = 0 to 119 {
    n139.grad[i210, 0] := (n139.grad[i210, 0] - n142.grad[i210, 0]);
  }
  for i211 = 0 to 119 {
    mlp.grad[i211, 0] := (mlp.grad[i211, 0] + (expectation[i211, 0] * n139.grad[i211, 0]));
  }
  for i212 = 0 to 119 {
    b3.grad[0] := (b3.grad[0] + mlp.grad[i212, 0]);
  }
  for i213 = 0 to 119 {
    n135.grad[i213, 0] := (n135.grad[i213, 0] + mlp.grad[i213, 0]);
  }
  for i214 = 0 to 119 {
    for i215 = 0 to 15 {
      w3.grad[0, i215] := (w3.grad[0, i215] + (n135.grad[i214, 0] * n133[i214, i215]));
    }
  }
  for i216 = 0 to 119 {
    for i217 = 0 to 15 {
      n133.grad[i216, i217] := (n133.grad[i216, i217] + (w3[0, i217] * n135.grad[i216, 0]));
    }
  }
  for i218 = 0 to 119 {
    for i219 = 0 to 15 {
      n131.grad[i218, i219] :=
        (n131.grad[i218, i219] + (n133[i218, i219] > 0.0 ? n133.grad[i218, i219] : 0.0));
    }
  }
  for i220 = 0 to 119 {
    for i221 = 0 to 15 {
      b2.grad[i221] := (b2.grad[i221] + n131.grad[i220, i221]);
    }
  }
  for i222 = 0 to 119 {
    for i223 = 0 to 15 {
      n129.grad[i222, i223] := (n129.grad[i222, i223] + n131.grad[i222, i223]);
    }
  }
  for i224 = 0 to 119 {
    for i225 = 0 to 15 {
      for i226 = 0 to 15 {
        w2.grad[i225, i226] := (w2.grad[i225, i226] + (n129.grad[i224, i225] * n127[i224, i226]));
      }
    }
  }
  for i227 = 0 to 119 {
    for i228 = 0 to 15 {
      for i229 = 0 to 15 {
        n127.grad[i227, i229] :=
          (n127.grad[i227, i229] + (w2[i228, i229] * n129.grad[i227, i228]));
      }
    }
  }
  for i230 = 0 to 119 {
    for i231 = 0 to 15 {
      n125.grad[i230, i231] :=
        (n125.grad[i230, i231] + (n127[i230, i231] > 0.0 ? n127.grad[i230, i231] : 0.0));
    }
  }
  for i232 = 0 to 119 {
    for i233 = 0 to 15 {
      b1.grad[i233] := (b1.grad[i233] + n125.grad[i232, i233]);
    }
  }
  for i234 = 0 to 119 {
    for i235 = 0 to 15 {
      n123.grad[i234, i235] := (n123.grad[i234, i235] + n125.grad[i234, i235]);
    }
  }
  for i236 = 0 to 119 {
    for i237 = 0 to 15 {
      for i238 = 0 to 1 {
        w1.grad[i237, i238] := (w1.grad[i237, i238] + (n123.grad[i236, i237] * input[i236, i238]));
      }
    }
  }
  /* end */
  /* end */