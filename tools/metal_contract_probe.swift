// Can MSL source pin floating-point contraction off?  Yes: a pragma does, and MTLCompileOptions
// has no property for it.  gh-ocannl-1836; the recorded verdict lives in
// docs/agent-notes/backend-precision-and-simd.md (the contraction-pin bullet).
//
// Standalone on purpose: no OCANNL, no OCaml switch.  It compiles MSL source at run time through
// newLibraryWithSource, as the Metal backend does, so it answers a question about the runtime
// compiler of the macOS it runs on.  Build and run (macOS 15 or later: it sets mathMode):
//
//   swiftc -O tools/metal_contract_probe.swift -o /tmp/metal_contract_probe && /tmp/metal_contract_probe
//
// The probe is (a * b) + c with a = b = 1 + 2^-12 and c = -(1 + 2^-11), read from a buffer so
// nothing folds at compile time.  a * b = 1 + 2^-11 + 2^-24 is a tie at f32 and rounds to even,
// 1 + 2^-11, so the serial two roundings give 0, and a fused multiply-add gives 2^-24.  Each
// kernel computes it twice: in one statement, and split over two (p = a * b; p + c).
//
// Its EXIT STATUS is the answer: 0 only if every contraction-off spelling returns 0 in both forms,
// and an explicit fma(a, b, c) still returns 2^-24, under each listed language version and math
// mode.  The bare kernel's values are reported, not judged.  On mac-studio (M4 Max, macOS 27.2.1)
// the bare one statement is fused in every mode, and the two statements are fused only under
// Relaxed and Fast.  A change there is news for the agent note, not a failure.
import Foundation
import Metal

let spellings: [(name: String, top: String, block: String)] = [
  ("bare", "", ""),
  ("METAL fp math_mode(safe), file scope", "#pragma METAL fp math_mode(safe)", ""),
  ("METAL fp contract(off), file scope", "#pragma METAL fp contract(off)", ""),
  ("METAL fp contract(off), kernel body", "", "#pragma METAL fp contract(off)"),
  ("clang fp contract(off), file scope", "#pragma clang fp contract(off)", ""),
  ("clang fp contract(off), kernel body", "", "#pragma clang fp contract(off)"),
  ("STDC FP_CONTRACT OFF, file scope", "#pragma STDC FP_CONTRACT OFF", ""),
]
// The spellings the exit status judges: each must give the two-rounding 0 in both forms.
let pins: Set<String> = Set(spellings.map { $0.name }.filter { $0.contains("contract(off)") || $0.contains("FP_CONTRACT") })

func source(top: String, block: String, explicitFma: Bool) -> String {
  return """
    #include <metal_stdlib>
    using namespace metal;
    \(top)
    kernel void probe(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
      \(block)
      float a = in[0], b = in[1], c = in[2];
      out[0] = \(explicitFma ? "fma(a, b, c)" : "(a * b) + c");
      float p = a * b;
      out[1] = p + c;
    }
    """
}

enum Mode: String, CaseIterable { case safe, relaxed, fast }
let versions: [(String, MTLLanguageVersion)] = [("3.1", .version3_1), ("3.2", .version3_2)]

guard let dev = MTLCreateSystemDefaultDevice(), let queue = dev.makeCommandQueue() else {
  FileHandle.standardError.write("metal_contract_probe: no Metal device\n".data(using: .utf8)!)
  exit(2)
}
FileHandle.standardError.write(
  "device: \(dev.name), \(ProcessInfo.processInfo.operatingSystemVersionString)\n".data(using: .utf8)!)

let a: Float = 1 + Float(sign: .plus, exponent: -12, significand: 1)
let c: Float = -(1 + Float(sign: .plus, exponent: -11, significand: 1))
var input: [Float] = [a, a, c]
let inBuf = dev.makeBuffer(bytes: &input, length: 12, options: .storageModeShared)!
let outBuf = dev.makeBuffer(length: 8, options: .storageModeShared)!
let fused = Float(sign: .plus, exponent: -24, significand: 1)

func run(_ src: String, _ version: MTLLanguageVersion, _ mode: Mode) throws -> (Float, Float) {
  let opts = MTLCompileOptions()
  opts.languageVersion = version
  switch mode {
  case .safe: opts.mathMode = .safe
  case .relaxed: opts.mathMode = .relaxed
  case .fast: opts.mathMode = .fast
  }
  opts.mathFloatingPointFunctions = .fast  // the backend's modern split: Safe plus fast functions
  let lib = try dev.makeLibrary(source: src, options: opts)
  let pso = try dev.makeComputePipelineState(function: lib.makeFunction(name: "probe")!)
  let cb = queue.makeCommandBuffer()!
  let enc = cb.makeComputeCommandEncoder()!
  enc.setComputePipelineState(pso)
  enc.setBuffer(inBuf, offset: 0, index: 0)
  enc.setBuffer(outBuf, offset: 0, index: 1)
  let one = MTLSize(width: 1, height: 1, depth: 1)
  enc.dispatchThreads(one, threadsPerThreadgroup: one)
  enc.endEncoding()
  cb.commit()
  cb.waitUntilCompleted()
  let out = outBuf.contents().bindMemory(to: Float.self, capacity: 2)
  return (out[0], out[1])
}

func show(_ x: Float) -> String { x == 0 ? "0" : (x == fused ? "2^-24" : "\(x)") }

var failures = 0
for (vname, version) in versions {
  for mode in Mode.allCases {
    for s in spellings {
      let label = "MSL \(vname) \(mode.rawValue) [\(s.name)]"
      do {
        let (one, two) = try run(source(top: s.top, block: s.block, explicitFma: false), version, mode)
        let judged = pins.contains(s.name)
        let ok = !judged || (one == 0 && two == 0)
        if !ok { failures += 1 }
        print("\(label): one statement \(show(one)), two statements \(show(two))"
          + (judged ? (ok ? "  ok" : "  FAIL: contracted") : "  (reported)"))
        if s.name == "METAL fp contract(off), file scope" {
          let (f, _) = try run(source(top: s.top, block: s.block, explicitFma: true), version, mode)
          if f != fused { failures += 1 }
          print("\(label) explicit fma(a, b, c): \(show(f))" + (f == fused ? "  ok" : "  FAIL: not fused"))
        }
      } catch {
        failures += 1
        print("\(label): COMPILE ERROR \(error)")
      }
    }
  }
}
print(failures == 0 ? "metal_contract_probe: every pin held" : "metal_contract_probe: \(failures) failure(s)")
exit(failures == 0 ? 0 : 1)
