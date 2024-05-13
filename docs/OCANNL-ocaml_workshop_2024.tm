<TeXmacs|2.1.4>

<style|article>

<\body>
  <doc-data|<doc-title|OCANNL optimization framework>|<doc-subtitle|Tensor
  shape inference, concise notation, multidevice
  runtime>|<doc-author|<author-data|<author-name|Šukasz
  Stafiniak>|<\author-note>
    Since April 2024, <hlink|<with|font-family|tt|<with|color|orange|a><with|color|blue|hrefs>>|https://ahrefs.com/>
    sponsors Šukasz's work on OCANNL.
  </author-note>>>>

  <abstract-data|<abstract|OCANNL is a Deep Learning framework with
  first-order automatic differentiation (aka. backprop) that implements low
  level backends, puts emphasis on shape inference and concise notation,
  supports multiple devices parallelism. Currently, at the core OCANNL
  provides explicit compilation and synchronization.>>

  <section|Powerful tensor shape inference>

  The most distinctive aspect of OCANNL is its row-based shape inference. A
  tensor shape in OCANNL is composed of three rows of axes: batch, input and
  output. In the underlying n-dimensional array implementation of tensors,
  inputs are innermost, but we use a types-inspired syntax:
  \Pbatch\|input<math|\<rightarrow\>>output\Q (or
  \Pinput<math|\<rightarrow\>>output\Q, \Pbatch\|output\Q, \Poutput\Q), where
  \Pbatch\Q, \Pinput\Q, \Poutput\Q are axis entries (an axis entry is either
  comma-separated or an individual character).

  The constraints we derive to perform shape inference are subtyping
  relations (inequalities) between rows. The subtyping relation accounts for
  <with|font-shape|italic|broadcasting>: matching dimension-1 angainst a
  greater dimension, and allowing for more axes in a row. A dimension
  specifies a single axis in a shape. The constraints include both dimension
  variables and row variables. Uniquely, the row variables in OCANNL are
  embedded: a row specification consists of innermost axes, an optional row
  variable, and optional outermost axes that come to the left of the row
  variable in the solution. The constraint solver solves both equations and
  inequalities between pairs of rows and dimensions.

  In OCANNL, a <with|font-shape|italic|shape logic> specification serves two
  purposes: it is an input to shape inference, but, once the shapes are
  inferred, it is also an input to <with|font-shape|italic|projections
  inference>. Projections contribute to the semantics of an operation by
  specifying how the tensors should be indexed. Currently, most aspects of a
  shape specification can be expressed using a syntax resembling the
  <with|font-shape|italic|einsum notation>, but much more general. For unary
  operations the syntax is <math|RHS\<Rightarrow\>LHS>, and for binary
  operations <math|RHS1;RHS2\<Rightarrow\>LHS>, where
  <math|RHS,LHS,RHS1,RHS2> are shape specifications. The syntax of a named
  row variable is <verbatim|..v..> for name <verbatim|v>, and an un-named row
  variable <verbatim|...> stands for a row variable named <verbatim|batch>,
  <verbatim|input>, or <verbatim|output> depending on where <verbatim|...>
  appears. To give some concrete examples, tensor multiplication (matrix
  multiplication generalized to multiple axes), would be an operation with
  operator <verbatim|*>, accumulator <verbatim|+>, and shape logic:

  <\verbatim-code>
    ...\|..args..-\<gtr\>...; ...\|...-\<gtr\>..args.. =\<gtr\>
    ...\|...-\<gtr\>...
  </verbatim-code>

  When no additional constraints are put on them, variables that appear on
  the RHS only (to the left of <verbatim|=\<gtr\>>), whether row or
  dimension, have their corresponding axes reduced (e.g. summed over) by the
  operation. Reducing (summing out) of the leftmost batch axis would be:

  <\verbatim-code>
    s...\|...-\<gtr\>... =\<gtr\> ...\|...-\<gtr\>...
  </verbatim-code>

  With an additional constraint on <verbatim|s>, this also represents a slice
  at a position of the leftmost batch axis. When numbers appear as dimensions
  in a specification, they provide axis positions. For example, reducing
  (e.g. summing) all entries of a tensor at 0-indexed position 1 of outermost
  output axis and 2 of innermost output axis:

  <\verbatim-code>
    ...\|...-\<gtr\>1...2 =\<gtr\> 0
  </verbatim-code>

  For pragmatic reasons, shape inference is organized into stages and makes a
  few heuristic assumptions at later stages. OCANNL also supports dimension
  labels (but not labels for selecting axes).

  \;

  <section|Code: representation, notation, optimization>

  Computation in OCANNL is organized into a declarative layer of
  <with|font-shape|italic|tensor> expressions, on top of an imperative layer
  of <with|font-shape|italic|tensor node> assignments. Both layers come with
  PPX extension points: <verbatim|%op> (operations) for tensor expressions,
  and <verbatim|%cd> (code) for assignments. A tensor comprises a value node,
  a gradient node, <with|font-shape|italic|forward> assignments and
  <with|font-shape|italic|backprop> assignments. Tensor operations notation
  fetaures <with|font-shape|italic|parameter punning> (strings become
  let-bindings of tensors) and inline output dimensions specification. Full
  example of a Multi Layer Perceptron with 2 hidden layers and Rectified
  Linear Unit non-linearity <verbatim|(?/)>, defining tensors <verbatim|b1>,
  <verbatim|w1>, <verbatim|b2>, <verbatim|w2>, <verbatim|b3>, <verbatim|w3>,
  and a tensor-returning function <verbatim|mlp>:

  <\verbatim-code>
    let%op mlp x =

    \ \ "b3" + ("w3" * ?/("b2" hid_dim + ("w2" * ?/("b1" hid_dim + ("w1" *
    x)))))
  </verbatim-code>

  Code notation for a Stochastic Gradient Descent update for a single
  parameter <verbatim|p>, where <verbatim|learning_rate> is a tensor, e.g.
  can undergo rate decay, <verbatim|(!.)> embeds a float as a
  tensor:<\footnote>
    Algorithm from <hlink|https://github.com/tinygrad/tinygrad|https://github.com/tinygrad/tinygrad/blob/master/tinygrad/nn/optim.py>
  </footnote>

  <\verbatim-code>
    let sgd_one ~learning_rate ~momentum ~weight_decay ~nesterov p =

    \ \ [%cd "pg" =: p.grad + (!.weight_decay *. p);

    \ \ \ \ \ \ \ if Float.(momentum \<gtr\> 0.0) then (

    \ \ \ \ \ \ \ \ \ "b" =: (!.momentum *. b) + pg;

    \ \ \ \ \ \ \ \ \ if nesterov then pg =+ !.momentum *. b else pg =: b);

    \ \ \ \ \ \ \ p =- learning_rate *. pg]
  </verbatim-code>

  Once the user collects the assignments of a desired routine, they are
  translated to a C language-like representation, interpreted to decide which
  tensor nodes should be <with|font-shape|italic|virtual>, computations of
  virtual tensor nodes (e.g. forward for value nodes and backprop for
  gradient nodes) are inlined, the code is simplified wrt. mathematical
  identies, and passed to a backend that the user selected.

  <section|Code execution: backends, devices, synchronization>

  In OCANNL, a tensor node is an \Pidentity\Q shared across the \Pfrontend\Q
  (OCaml runtime aka. <with|font-shape|italic|host>), backends and their
  devices \U the same tensor node can correspond to arrays of numbers in
  multiple places. A tensor node can be <with|font-shape|italic|merged> \U
  pointwise reduced \U across instances. Tensor nodes can be:

  <\itemize>
    <item><with|font-shape|italic|Virtual> \U without corresponding arrays
    and absent from the generically optimized code.

    <item><with|font-shape|italic|Local> \U an array is cached for the
    duration of a computation but not persisted across calls to compiled
    functions.

    <item><with|font-shape|italic|On-device> \U an array is stored on the
    devices that compute with it and persisted across function calls. It is
    available for merging across devices (for devices that support merging /
    P2P), but not for visualization or storing to disk.

    <item><with|font-shape|italic|Hosted> \U the array is stored in a
    globally addressable memory, in addition to on devices where it is
    computed with (or as part of one of them, if ``hosting on device", or
    only on the host and not on devices, for some backends).
  </itemize>

  Tensor nodes have each a numeric precision \U OCANNL supports mixed
  precision computing.

  A backend may optionally compile a routine in a relocatable way, to save on
  compilation times, and later link it to particular device contexts.
  Currently, OCANNL assumes that devices have queues of length 1: scheduling
  a new task blocks till the previous task is scheduled. CPU backends in
  OCANNL offer CPU cores as separate devices. OCANNL offers a round-robin
  scheduler for multi-device data parallel training, with \Plogarithmic\Q
  merging of gradients.

  OCANNL is integrated with the package <hlink|ppx_minidebug|https://github.com/lukstafi/ppx_minidebug>,
  including tracing device execution.

  OCANNL was inspired by Andrej Karpathy's
  <hlink|micrograd|https://github.com/karpathy/micrograd>, motivated by
  projects <hlink|tinygrad|https://github.com/tinygrad/tinygrad/> in Python,
  and very recently <hlink|Luminal|https://github.com/jafioti/luminal> in
  Rust. We will also study the project <hlink|llm.c|https://github.com/karpathy/llm.c>
  as the \Pgold standard\Q for optimized compilation results.
  Functionality-wise, OCANNL has barely started: as of May 2024, future work
  include providing GPU parallelization, a neural networks toolbox, a more
  convenient and powerful scheduling. The design of OCANNL might still
  significantly evolve.
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
    <associate|preamble|false>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1>>
    <associate|auto-2|<tuple|2|2>>
    <associate|auto-3|<tuple|3|2>>
    <associate|footnote-1|<tuple|1|2>>
    <associate|footnr-1|<tuple|1|2>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Powerful
      tensor shape inference> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Code:
      representation, notation, optimization>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|3<space|2spc>Code
      execution: backends, devices, synchronization>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>