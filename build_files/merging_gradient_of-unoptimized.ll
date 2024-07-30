
merging_gradient_of_b1 ():
  /* merging gradient of b1 */
  for i764 = 0 to 15 {
    b1.grad[i764] := (b1.grad[i764] + b1.grad.merge[i764]);
  }
  /* end */
merging_gradient_of_b2 ():
  /* merging gradient of b2 */
  for i766 = 0 to 15 {
    b2.grad[i766] := (b2.grad[i766] + b2.grad.merge[i766]);
  }
  /* end */
merging_gradient_of_b3 ():
  /* merging gradient of b3 */
  b3.grad[0] := (b3.grad[0] + b3.grad.merge[0]);
  /* end */
merging_gradient_of_w1 ():
  /* merging gradient of w1 */
  for i769 = 0 to 15 {
    for i770 = 0 to 1 {
      w1.grad[i769, i770] := (w1.grad[i769, i770] + w1.grad.merge[i769, i770]);
    }
  }
  /* end */
merging_gradient_of_w2 ():
  /* merging gradient of w2 */
  for i773 = 0 to 15 {
    for i774 = 0 to 15 {
      w2.grad[i773, i774] := (w2.grad[i773, i774] + w2.grad.merge[i773, i774]);
    }
  }
  /* end */
merging_gradient_of_w3 ():
  /* merging gradient of w3 */
  for i776 = 0 to 15 {
    w3.grad[0, i776] := (w3.grad[0, i776] + w3.grad.merge[0, i776]);
  }
  /* end */