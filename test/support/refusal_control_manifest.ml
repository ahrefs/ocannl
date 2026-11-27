(** The explicit bridge from mechanically extracted scanner diagnostics to their assigned permanent
    control suites. A new diagnostic is absent here by default; the catalogue test holds this
    manifest equal to the extractor and holds every entry against the assigned golden.

    This table is inventory, not evidence. [print] emits a claim marker only when Verdict recorded a
    successful execution of the claim's diagnostic format, consuming each execution once, so
    identical claim formats need one execution apiece. It emits a direct-failure marker only when
    the refusal RAN ([standing]): its caught branch recorded the format through [observe_failure],
    or the control claim assigned below passed and a refused child run of the scanner printed a line
    that is the refusal's own. Refused lines are not consumed: two sites sharing one direct-failure
    format would share one line, which no scanner has today. Adding a row here therefore cannot make
    its own golden pass. *)

open Base
open Stdio

(* cc-fp-contract: not-executed -- catalogue entries quote scanner diagnostics *)

let raw_entries =
  [
    ( "explicit_library_deps.ml",
      [
        "[scanner-refusal:5d96a6af77f2e3cc3e176d98155165fa] action's variables resolves";
        "[scanner-refusal:a5d692f42ee189e1a52b3863f56276ba] action's variables copies";
        "[scanner-refusal:65eb18ba581fbe9366bfc0028974c962] nested-Dune environment resolves";
        "[scanner-refusal:19ebc49024b1c4e3db05242a22f8b0d5] nested-Dune environment leaves";
        "[scanner-refusal:79b83d87bc35a637a45c0dd6dc7479cf] explicit_library_deps";
        "[scanner-refusal:e2937f725c3dd416fbbbb047979dc15a] library declaration compiles";
        "[scanner-refusal:8261260fa2908a6e512581ef47939697] undeclared use compiles";
        "[scanner-refusal:bd4e1515861d31f929904e3deb2b0b7b] undeclared transitive library";
        "[scanner-refusal:36550c0b9c79bcf4a0384888ee112619] declared library's re-export";
        "[scanner-refusal:068ac0a43201ecb7fea5899eba10c386] runtime dependency compiles";
      ] );
    ( "unused_compiler_deps.ml",
      [
        "[scanner-refusal:38ed84dac52cdd93b3e3c36d807a9b78] csexp an unclosed";
        "[scanner-refusal:7b9f08df1830b83e15f81735eb806b0a] atom length without";
        "[scanner-refusal:8e4f5848655803e7e25d07ef21708b5d] atom runs past";
        "[scanner-refusal:a5b33c62232554fef4df88809fa612c6] csexp unexpected byte";
        "[scanner-refusal:46d0267ea0664bd9e2a413612e062656] csexp an unbalanced";
        "[scanner-refusal:4af5e799f2d3c8b89fbbe0264fb9a4da] dune describe workspace";
        "[scanner-refusal:cc8f7ce3ee8855d58ceda3c3058bb5ae] outside the recognised";
        "[scanner-refusal:de4b903cfa394b4049d2db6381615ab2] describe a library";
        "[scanner-refusal:18f29b05a5530dc0b19d1f70a9e46645] describe a library's";
        "[scanner-refusal:44bdc924085a33ef9a9fe15cbc3a9dfa] describe a library";
        "[scanner-refusal:e78a529971f9ec7b38bb24067becb24a] context other than";
        "[scanner-refusal:6b085283b3405e85ba3d11513cb884a8] outside the recognised";
        "[scanner-refusal:819aea460d5462662096e99432e67c9a] describe not one";
        "[scanner-refusal:6e0e71605968988d9c01bcfd8d28af8a] recognised unit definitions";
        "[scanner-refusal:dabba7b36e489ee77f7d116fc69bf1ef] malformed interface import";
        "[scanner-refusal:cb2bfd7df9e67d7e7bfae1edabb9ca51] could not inspect";
        "[scanner-refusal:cc150842e2a3378d147846073232fa3f] shared include directory";
        "[scanner-refusal:16f777ad18ff660f1debccb5fbf474fd] archive query for";
        "[scanner-refusal:53c0a1deae3afb540f685cce0e25b545] interface-only unit";
        "[scanner-refusal:ec195edbfb60eee2dbaf9bcc4c5938ea] stanza without a";
        "[scanner-refusal:584d40dd3eabb5585f6ced0fe671eb57] stanza whose stanzas";
        "[scanner-refusal:532ff6f33244b0d4e6bac5171d60e0b6] carries a libraries";
        "[scanner-refusal:63b88fa452bec0fdd00be66980960322] pform in libraries";
        "[scanner-refusal:712ec45e25069175f8164f1106fcc5fb] outside the recognised";
        "[scanner-refusal:1e9d3fc1767c7da9e48ecb3a1bb3f179] select arm holds";
        "[scanner-refusal:b70a2b3bb6d2686b6ec9fcc2d0778373] libraries element outside";
        "[scanner-refusal:9c87841ead2852ff3454a7426f94e5dc] the declared library";
        "[scanner-refusal:90c4db4dc0ddba1d876dfb79175075f8] no object directory";
        "[scanner-refusal:e654ceab66220252a649c7bb36aa0e49] object directory";
        "[scanner-refusal:4bb6aa436016609d636df4dbb21c88fd] unreadable typed tree";
        "[scanner-refusal:1705a50114bbcfdc7d549777efcf1348] no single library";
        "[scanner-refusal:1262ba0367f56ab63b0ef45185bb1257] control stanza must";
        "[scanner-refusal:1c8850699d8f9fec6290b9542a410741] an unreadable archive";
        "[scanner-refusal:e25a1cf286ceb521ee4eb7746aad5229] archive tool producing";
        "[scanner-refusal:ff2924c3a9869dd3bc7f7d72acd8ead1] malformed archive interface";
        "[scanner-refusal:f684c2c391ba5d7246a2468a4d4584ca] failed archive query";
        "[scanner-refusal:38b0123ae5353ab2b03e006a018d2bc6] shared local library";
        "[scanner-refusal:8321fb2ab306d6d7f47b17136d308f5d] unique dependency-minimal archive";
        "[scanner-refusal:9444a57b2ad4731e29ce408670e73e59] cyclic interface-only owners";
        "[scanner-refusal:afa29a41b9bcf5f2edb420167b2557b1] consumer-only interface import";
        "[scanner-refusal:e465b31053262fa064a9c23ff3892b2c] dependency's archive definition";
        "[scanner-refusal:52ed7f2b3e43e26453e5f6c31ebb5965] co-located archive-less dependency";
        "[scanner-refusal:ffa4bc91bdd9e0c6519d47ab6e078f3c] csexp reader refuses";
        "[scanner-refusal:b2e2da4cfdb3763209d5bb0bf037f3eb] description that is";
        "[scanner-refusal:cb9d50292bf8e1411a78fe56d95365d7] description entry of";
        "[scanner-refusal:f9ea26ffa524f825f454965f1ef68877] control library description";
        "[scanner-refusal:e5da8bbf11bdc53c4c4f1d6394be7f88] outside the recognised";
        "[scanner-refusal:d0dd9b0705f1359338d24f701064b435] context other than";
        "[scanner-refusal:72246cd3e9e00d9ffa244cdd2bb49513] library without its";
        "[scanner-refusal:92206b092da67b9faa6ae7b73171723f] library whose requires";
        "[scanner-refusal:19d2ba86f2b359e5c32a81e81062f7c5] without its requires";
        "[scanner-refusal:c0761275e39346be1c8837a5a85d98b6] a stanza without";
        "[scanner-refusal:7a60a99f8c898d07163567eeff851cbc] another stanza carrying";
        "[scanner-refusal:1987016d8fdca3b45ba0d604394f8ec5] include stanza hiding";
        "[scanner-refusal:9a0fb44910c66f1c9dcd5eed94766764] while its declaration";
        "[scanner-refusal:973eea87de432c77fdffdc995b8c64f3] row whose declaration";
        "[scanner-refusal:8e3c17c0968ce3e98f1d8ad0b89c6718] libraries is refused";
        "[scanner-refusal:c319818e5b162fbc67bb85019fff6f99] libraries element of";
        "[scanner-refusal:0a4b1d22d540d21f605c894b621ea1c1] outside the recognised";
        "[scanner-refusal:ab518590d10da57cedc8d0717e63af61] holds is refused";
        "[scanner-refusal:828116468b6f2911e974fddf4fe601d8] declared name nothing";
        "[scanner-refusal:f0f1587aca0a7afae0b83e27bed22b38] enabled_if";
        "[scanner-refusal:910c1650b37b93c2cd7917497f236fb2] enabled_if";
        "[scanner-refusal:7398f68cb579d4b59ac9ce927f234276] object directory holding";
        "[scanner-refusal:b1784c07fa1dac619848cd1b4a9d64f0] object directory holding";
        "[scanner-refusal:69ce705759093b5f97c6cc8e0413a500] description dune refuses";
        "[scanner-refusal:8c24881b343b90be6f9574bf281e5045] or multiple libraries";
        "[scanner-refusal:80993f82fb1569705822b50a7436584e] rejects another refusal";
        "[scanner-refusal:351ce2a461cb37393c61069ee2649d96] unused_compiler_deps";
        "[scanner-refusal:36778054d93aa08eac389989017aed7a] DUNE_SOURCEROOT";
        "[scanner-refusal:044e249871007719b6bc46d149189c1d] synthetic tool calls";
        "[scanner-refusal:22c08112ff7b660af6dab49542313b14] compiler-libs.common";
        "[scanner-refusal:5c6002f7f040eddc8696d03b9a9bb9e2] the workspace description";
        "[scanner-refusal:448291c7f6134ac8fadedd8e1a4f787a] declares the library";
        "[scanner-refusal:348bd3c284e68555c5d779b004a5cb5f] compiler-reaching or test-support";
        "[scanner-refusal:76d536378a6e503a67ce436cfd76938d] fixture directory's stanzas";
        "[scanner-refusal:67a884b3a6347974dc0b1b43122c4c21] derived as compiler-reaching";
        "[scanner-refusal:a0a5de6bdc283aef59519729356595e4] test-support fixture finding";
        "[scanner-refusal:4a372cdf7b39da35fd50d7a96957b75f] its expected libraries";
      ] );
    ( "markdown_links.ml",
      [
        "[scanner-refusal:2f2dfef90107c2fbc43b20d212add0ef] Markdown corpus includes";
        "[scanner-refusal:63555f8686b2d8399a5909d32c43838b] documented symlink alias";
        "[scanner-refusal:497f7864b862de90bf699cb29ebc7395] recognised relative Markdown";
      ] );
    ("doc_examples_extract.ml", []);
    ("ll_test_ratchet.ml", []);
    ( "premise_ratchet.ml",
      [
        "[scanner-refusal:b58a95cec37dfce6d01a2e4de53038d6] test source parses";
        "[scanner-refusal:e74f4879325b87eebe84c1ce30d37ed2] premise debt has";
        "[scanner-refusal:b35b45534098c6165bc2d50aab63313c] contains exactness premises";
      ] );
    ( "exact_partials_ratchet.ml",
      [
        "[scanner-refusal:ba759c72b65ab51d06be7e24325abbb2] every test source";
        "[scanner-refusal:958e88eddd55da1a156f3795cd586c78] beside a derivation";
        "[scanner-refusal:b30b6b3be7749ff984d2979f5a8fe292] arrayjit package's test";
        "[scanner-refusal:392052035d276f11311768815316b580] every sentence claiming";
        "[scanner-refusal:fa85b1960c6367c9216c627d02ede0e7] every exemption row";
      ] );
    ( "fp_contract_pin_scan.ml",
      [
        "[scanner-refusal:b2b66576ec487b82b65d3d2a3c7fb584] sentinel must open";
        "[scanner-refusal:5fae6b89f7d8258826c7189aafba5a40] expected <kind> <reason>";
        "[scanner-refusal:9970d742a14c33910983647a40f64148] unknown kind";
        "[scanner-refusal:897ec4f61958b8709e5a89603692821a] the reason needs";
        "[scanner-refusal:0477ef9a58ec9e25b75d9c0f1a34c306] bitwise claim for";
        "[scanner-refusal:21592137723d61ef8766bb4585efb5d1] marker drop the";
        "[scanner-refusal:7be6e2776283b7cf291a429499bfe007] than one marker";
        "[scanner-refusal:4c8ebb0a854f491b5b3bf153cb8e6e1f] OCANNL_CC_BACKEND_FP_CONTRACT=off";
        "[scanner-refusal:e15c6d30a70f479a8a6380750aa59042] does not parse";
        "[scanner-refusal:749f02c83c0859bd8145401cc5a4a5cb] scanner raised";
        "[scanner-refusal:106879e8d84d7b106e818ac1ddc61a4f] listed as pinned";
        "[scanner-refusal:777651b1b86abfe512e2d3a3f485ce6d] marked claim is";
        "[scanner-refusal:0917e598d504b434f3e3c9567742fcc6] claim with neither";
        "[scanner-refusal:9d3c109a283ee3d6803e356b583bdff2] claim in capitals";
        "[scanner-refusal:3ef51a84cfe8ca3b03690fcda8b63c21] to another mode";
        "[scanner-refusal:02d106e7052951bcf97aada0e6bd4f13] Utils.state_config";
        "[scanner-refusal:08304716a8de10400315dadefbfc01d0] Utils.state_config";
        "[scanner-refusal:6b3c897a51c7aa7f19a5aee3aa606bb8] Utils.state_config";
        "[scanner-refusal:19b74f64556e25424a61189b12a8d9c1] Utils.state_config";
        "[scanner-refusal:a1474866d87b39b20580a308ff625d49] Utils.unstate_config";
        "[scanner-refusal:19ba2d6c29ec6da6833b4f6102b96167] an unknown kind";
        "[scanner-refusal:6a15033dd60a8451f9ff4ba97f4280cb] one-word reason is";
        "[scanner-refusal:be5184012422a01273ca348715955f05] sentinel that does";
        "[scanner-refusal:b79e2ebc86bfefafc47281afbf5f0550] markers are refused";
        "[scanner-refusal:3237007207743bc53ec1c1951d1214b7] together are refused";
        "[scanner-refusal:b52b141cf7d13778997e41d81205de03] claim is refused";
        "[scanner-refusal:ffc0217024cc9300c720ee267dd0fde3] separator is refused";
        "[scanner-refusal:640314079c325f8264eadc3ee11e14a7] the expected diagnostic";
        "[scanner-refusal:61cc31cefcfa4e62f663ac4758af255a] source that does";
        "[scanner-refusal:e9713e59d79f302d02300babf9ebd87c] is refused separately";
        "[scanner-refusal:e28d8e210f8906b6927a78032d5a7f50] case the word";
        "[scanner-refusal:5af7923d754adb0ac1030df866493123] every test source";
        "[scanner-refusal:b30b6b3be7749ff984d2979f5a8fe292] arrayjit package's test";
        "[scanner-refusal:3cf4cde03d1a75d1846d0bbad2327014] the mode in-process";
        "[scanner-refusal:cd11cea99124c2fa6c076ec8734ee25f] listed source carries";
        "[scanner-refusal:9bedd04370c001da7559696548d4a37b] cc_backend_fp_contract=off";
      ] );
    ("refused_action_streams.ml", []);
    ("operand_key_ratchet.ml", []);
    ( "tensorized_census_ratchet.ml",
      [
        "[scanner-refusal:4656151bdc35a3727a1bea78aa31e974] census-judging test declares";
        "[scanner-refusal:e553a74d077e12582953b27756595d9e] executed tensorized parity";
        "[scanner-refusal:acf79e9d5f580995dafd055521de641f] Context.run";
      ] );
    ("provenance_tag_inventory.ml", []);
    ( "resource_fault_inventory_scan.ml",
      [
        "[scanner-refusal:c85bb67d5d4a65e371321bd39eff4686] derived fault-injection point";
        "[scanner-refusal:32d44718d7ebdf4c4f7d6d2d289ff1cd] inventory table row";
        "[scanner-refusal:6cd996e65b0a28170b37d4a74708cfee] every fault-injection point";
      ] );
    ( "open_node_scan.ml",
      [
        "[scanner-refusal:d7ba33c928dde8cf6b59d769ad3a0b85] does not parse";
        "[scanner-refusal:53596805b743d859cfe3bcff6c3f4cef] open-node reader raised";
        "[scanner-refusal:734ae927ae65fe211d322172ca5b596c] among the sources";
        "[scanner-refusal:ba81e42164f42036a795a1c48ab2f6c2] tree matches nothing";
        "[scanner-refusal:69cf046689c576d9dca494c4b411d591] Tnode.create";
        "[scanner-refusal:aad2a5fc0b2e347174ab90714113a6f6] tensor.ml";
      ] );
    ( "sweep_owner_scan.ml",
      [
        "[scanner-refusal:ccfbd7919a1ffc7c1e052d9d49911b39] does not parse";
        "[scanner-refusal:69de7c57450df323441e9dc5d85110a3] outside-sweep owner reader";
        "[scanner-refusal:d041f52ef5edce112db1ae82bf45699f] tree matches nothing";
        "[scanner-refusal:13a68f9fac09b107d35175bdfad613d4] cc_backend_trace_name";
        "[scanner-refusal:9fd940ff3d9f45509ae4d01b3bc7efdb] resolves a plain-run";
      ] );
    ( "debug_name_scan.ml",
      [
        "[scanner-refusal:fe40918919ee07d96a2c92a926ae0b28] does not parse";
        "[scanner-refusal:035b75b8ce8b0d0f0b1b390267c227e7] debug-name reader raised";
        "[scanner-refusal:9168a34c28bab17307d7535ed07d097f] tree matches nothing";
        "[scanner-refusal:cdb9f02e250554d24674371724942ae4] test_tnode_namespaces'";
        "[scanner-refusal:693fe71b6aeb6d44bd88787308cf5017] cc_kernel_outline's";
        "[scanner-refusal:94cdb1a16d792be58f07d154224626df] scope_over_materialized's";
        "[scanner-refusal:f4f4d81c0884b6209eccd072274b494d] rewind_to_last_closure's";
        "[scanner-refusal:faf8bd39d9f3358220e041edafe252a2] Session_error";
        "[scanner-refusal:c622584e09b7ad9b23646841eb389ca9] @message_only";
        "[scanner-refusal:3c688ea5897f696f0882bff06b550c6e] debug_name";
      ] );
    ( "label_dispatch_scan.ml",
      [
        "[scanner-refusal:7fe4eca3a2beb271d0627103af78040c] does not parse";
        "[scanner-refusal:1705e6ff4f79cfed2aa6028e5b60cadc] label-dispatch reader raised";
        "[scanner-refusal:20abdf23cc9b4767a5eb4f75bb006965] source under arrayjit/lib/";
        "[scanner-refusal:298ad457cb53221b4d5bf26c57db10cb] online_softmax's";
        "[scanner-refusal:a134c194b11e19f42951ac9ae0496f39] scan reads backends'";
        "[scanner-refusal:5478be505fa459772de23c278bbd2980] ~backend_name";
        "[scanner-refusal:1953375f4f82df9f942fae5674bb2e4d] @label_read";
        "[scanner-refusal:753c4e61524edee33f56c2a0183bca9f] memory_budget's";
        "[scanner-refusal:17bc0c94b9ab448cbc75a7f5df828695] low_level's";
      ] );
    ( "render_decline_inventory.ml",
      [
        "[scanner-refusal:d5f512590dad4114be7fccd7fcad252d] control";
        "[scanner-refusal:93267baf53efdf52a09d9a689b122ee2] render-time typed declines";
        "[scanner-refusal:a981180a06d8df6fd4de07659591ded7] arrayjit/lib source parses";
        "[scanner-refusal:088c4ae45b34730d522c2b57bb9d8963] render-time typed declines";
        "[scanner-refusal:dd8e45f07b3a48f23ebf2b744ace13cf] every entry's classification";
        "[scanner-refusal:820965bd005837e9b9226c12de25898a] paired or render-only";
        "[scanner-refusal:d5a3d0a93c03dd4c310fea6768cacf86] Backend_codegen";
        "[scanner-refusal:4f88c489b62274a26eeacde0d502244e] Cause_at";
        "[scanner-refusal:15a7945bea16c6bb1aa0978c1e19745c] op_legality";
        "[scanner-refusal:b8d04b43e856d40dadc6df2ae5eb1951] an oracle constructor";
      ] );
    ( "atomic_file_rename_scan.ml",
      [
        "[scanner-refusal:7191f5e19a3640926ad61aeb2d7a349e] source-generation input has";
        "[scanner-refusal:c045a29a4e2f18a646143f07d6ec923b] missing corpus section";
        "[scanner-refusal:b5717d7e9eb6f55e37e9f5f57d114fc1] present and nonempty";
        "[scanner-refusal:5b91d4ebbeebe59d5615dd99fe73edfd] unexpected corpus section";
        "[scanner-refusal:bccd646d58b5010b1fe4eee6abe7eb53] corpus arguments must";
        "[scanner-refusal:fb72b56980e9afc888059f90e9734599] arguments the rule's";
        "[scanner-refusal:20f449385ebca8a7a25b2d306c8e0f44] scan cannot vouch";
        "[scanner-refusal:125c419015c4e6332b0fa19c8e6a1762] rename scan raised";
        "[scanner-refusal:77ea18b8b8167a5861a77a9a6bbe45bf] unrelated rename spelling";
        "[scanner-refusal:298ab0be4c86a0dabfaeeddc5e67bd29] contributes its derived";
        "[scanner-refusal:dfeaf03eef363cf0944237d1f754cb5a] generated-source negative control";
        "[scanner-refusal:16447aab0e906595f837aaa44096abbc] Atomic_file";
        "[scanner-refusal:698fae6b12a0d52a71f789bc962a882b] raw rename exemption";
        "[scanner-refusal:ed6060063668d409574488fcb7fc8897] exemption matches exactly";
      ] );
    ( "agents_md_size.ml",
      [
        "[scanner-refusal:9b96dba1c5b1fe8c4a37c837ecca6ddd] empty or misrouted";
        "[scanner-refusal:b9a291d865e79e74582ef22daf02fd3a] AGENTS.md";
        "[scanner-refusal:2bd517cf99304ae6d8358f4bfc6d3995] synthesized guide of";
        "[scanner-refusal:6e5c887df5f7f65a45a232ef22d04e62] synthesized guide one";
        "[scanner-refusal:6f380ec6130d7d24a10cfd39521bdaec] synthesized guide at";
        "[scanner-refusal:2c8e54e9bdb6518252759d38c07973aa] synthesized guide one";
        "[scanner-refusal:c32f3dd124b270ba497cf74ca1bc7e77] synthesized guide of";
        "[scanner-refusal:87f68dbf665f63c6666dc15bdfe9bc9b] longest-line advice is";
        "[scanner-refusal:fbbca570493f5d6fe407e047220ac6fa] longest-line advice keeps";
      ] );
    ( "agent_notes_structure.ml",
      [
        "[scanner-refusal:bc4968cf04b3b7f3212afef877977199] handed the notes";
        "[scanner-refusal:2ac555c09ba6fe90fc6321fecba16cab] the notes' bullets";
        "[scanner-refusal:3b1b44bc902f09eaba8469a50662c285] tools scripts' pointers";
        "[scanner-refusal:94abf9ad59c6aa58e4608e31b0c1481a] sources' string literals";
        "[scanner-refusal:ec487246d0ff9ad951333a2c56d4a990] names is reported";
        "[scanner-refusal:ed552c5b3e09a3046aa9cce5257cc148] every finding carries";
        "[scanner-refusal:348bb1f1fa0be73fb4796a31a76abaad] each flagged bullet";
        "[scanner-refusal:ec62d0a7d728c7ab15a2c9b8608c6404] every exemption still";
        "[scanner-refusal:91d29d400185d4fe574f5deb83d48157] every pointer-fixture exemption";
      ] );
    ( "changelog_unreleased_scan.ml",
      [
        "[scanner-refusal:29e846ac8743ceba34a4cdcd5c90a431] CHANGES.md";
        "[scanner-refusal:2a02d6286a27aef4f8d7e61e279f8c4e] a two-space continuation";
        "[scanner-refusal:0ec3ef6e76a1b717d0c1be40a6364fa2] every continuation line";
        "[scanner-refusal:8d06b043d3c8178f1e392a5d83d4e288] every Unreleased bullet";
        "[scanner-refusal:ea9d824f2fc10de3c98ef75465ca880b] bullet cites gh-ocannl-NNN";
        "[scanner-refusal:e2c181fc0e256bd7f1088f4936eb3f67] eighteen synthetic control";
        "[scanner-refusal:30ee8e64c0adfe9a46f60a7f898e7b70] synthetic four-line bullet";
        "[scanner-refusal:39d8ffc71b6ccd79bc151918c558c2b6] bullet whose fourth";
        "[scanner-refusal:c4aa67d124cb4749d1889edf6983c9a9] synthetic uncited bullet";
        "[scanner-refusal:23fceb1687bd50c5958f39301e78e77a] flags an unqualified";
        "[scanner-refusal:1f0ae7533580b5e4beb238c4275c1ab4] citation rule flags";
        "[scanner-refusal:bb0069493c829780cec5ca1d66feab8d] issue number running";
        "[scanner-refusal:6e22c3db47b0a6937bb1ab9c0855ebb5] citation rule flags";
        "[scanner-refusal:3ca66d41359dc51f6ab5f7ac84b8b907] citation rule flags";
        "[scanner-refusal:c9449faf44ed458347e1d657e626adee] issue prefix embedded";
        "[scanner-refusal:ba18f4595ed7faf639f772d898467c81] citation rule flags";
        "[scanner-refusal:d25d6e7f710421f3a575ba500564cfde] citation rule flags";
        "[scanner-refusal:62d1e4f42c52ac0924aa8f4d9c39f570] citation rule flags";
        "[scanner-refusal:591e045974c801797d7307f579cdfe66] quoted tag attribute";
        "[scanner-refusal:0d3e616cb68f4ff90f733dd0e2c4e994] citation rule flags";
        "[scanner-refusal:04232f9c257b6d2ecbf350399d42c1f0] citation rule flags";
        "[scanner-refusal:19ad9fb4c83fa6fb1dade7ec100c1fd6] span wrapped across";
        "[scanner-refusal:f517d12e9916c479313e517d39c992aa] citing a staging";
        "[scanner-refusal:ec0eb57bb1e3bf8c0c0fd7d145ab5b3e] citing a gh-ocannl";
        "[scanner-refusal:b58186611079f33f0c1982b767655695] refuses every shape";
        "[scanner-refusal:13ab4d5a9f0385a905c692cb883674ab] accepts every shape";
        "[scanner-refusal:d6e62794b51d807f87aec4370f3a751a] refuses a continuation";
        "[scanner-refusal:5bcdaf0ef54cbe80e0d80101fe648b77] section reader refuses";
        "[scanner-refusal:276a908bf7fcd396f224bf426067c796] Unreleased line counts";
        "[scanner-refusal:c4907873eb4e13a193737314f62d9d0b] the anchor's text";
        "[scanner-refusal:5b3f45ee369cc10ca978db8f30e90719] four-space-indented copy of";
        "[scanner-refusal:194c1e57cc9420634205a146c56e16d0] an anchor quoted";
        "[scanner-refusal:f3befecb42f5ce5ca62d20d3ec29b4e5] anchor quoted inside";
        "[scanner-refusal:8d85b5a872954cff5a8c20b7ad997075] backtick fence's info";
        "[scanner-refusal:b1060c3264fad2b26750763e08dbc86f] anchor inside released";
        "[scanner-refusal:05a94c42ce59cc69276bc3f89d99172e] unrelated closing tag";
        "[scanner-refusal:2b7eb419ed7900fbb86ba90fcfb6d542] hides nothing after";
        "[scanner-refusal:5992ccadeb90a0283af47dd6b29ab152] merely starting with";
        "[scanner-refusal:515b34b740d4146ebf521dfdde7f936b] is still structure";
        "[scanner-refusal:7ed0378e940ae971901b7ba2832b2902] with content after";
        "[scanner-refusal:809c8d3c58f68a68e3b81ccdd786a91d] terminator and hides";
        "[scanner-refusal:4ecb12e641c9ece3bcdf86b758da156d] lowercase cdata opener";
        "[scanner-refusal:1ccaa2c011552dd2a20a501a8a08a652] lowercase declaration opener";
        "[scanner-refusal:5275f7cebe83a0e8088068748c8f9ade] setext heading ends";
        "[scanner-refusal:897949d8f847215283e8799badf7f07f] comment opener inside";
        "[scanner-refusal:95468bc3bac3466d68db9d3b9602b3f2] level-4 subheading stays";
      ] );
    ( "backend_golden_family_scan.ml",
      [
        "[scanner-refusal:6e3c13cde68fdb8c13e4fe2e78b895d7] one backend position";
        "[scanner-refusal:f9d6efa26b74daefa5557b7b00f8a682] expected atom";
        "[scanner-refusal:5d618d2854245d7a364a03971b3a1b0d] expected atom";
        "[scanner-refusal:de53562eca7cfdcf8e2a137a77d44724] <member>.expected";
        "[scanner-refusal:8b581264b8bd140e3985e9f9bced6638] provenance target must";
        "[scanner-refusal:1c96d667ded2ef1c95b7780e45d29789] recorded-on backend";
        "[scanner-refusal:5ac098d53c612962b81e3afe44054bdd] provenance marker has";
        "[scanner-refusal:43131f24bfb3bb850487bf36adc98bb5] provenance marker reason";
        "[scanner-refusal:9dd518b47eeaa289c03164634c04f299] more than one";
        "[scanner-refusal:b488d4e27ab1c2ebf5ac694c7c7095fa] that references family";
        "[scanner-refusal:8a4d6f9fcf6429492277e482bfa47424] occurrence(s in the";
        "[scanner-refusal:6620d38a048422186913091223d863d9] parse dune structure";
        "[scanner-refusal:1407376649d9a0054747b260309b94b2] provenance marker names";
        "[scanner-refusal:8dd5f5e0febece68bce7c390ae336362] provenance marker target";
        "[scanner-refusal:dfe88f31831f61d09b8eecbf1fe2cd36] provenance marker records";
        "[scanner-refusal:69adad8dff9bfd72ab49a01d9ea8b4a8] contain exactly certification";
        "[scanner-refusal:fb5ea0107d1672c9e7c66baa76f18f9a] backend golden family";
        "[scanner-refusal:2fcd4d491da746e829dc7cec63905091] self-certifying and provenance";
      ] );
    ( "bin_smoke_membership_scan.ml",
      [
        "[scanner-refusal:9446215c7b3c2270741bbb1cd8300733] executable exactly once";
        "[scanner-refusal:75093564cde01f102b3c81d2ebde6ccd] synthetic public executable";
      ] );
    ( "generated_lock_scan.ml",
      [
        "[scanner-refusal:feb2a77e5687a83594725de6a658eb72] names the lock";
        "[scanner-refusal:5a695f9728903023648f3c3009d581c9] locks \226\128\166 field";
        "[scanner-refusal:1926b093c11bb0b478fa3697c4d721bf] names Generated and";
        "[scanner-refusal:43db180ec02405f5d8629c5c69f525d3] Generated.init";
        "[scanner-refusal:9de3b3fd1cd96a8d87e76bfa31534244] comment that does";
        "[scanner-refusal:d6feeef530479095718ac0f80b9bb9c6] carries a generated-lock";
        "[scanner-refusal:7b93ef6acc4c55c6949954343adf990e] generated-lock marker that";
        "[scanner-refusal:a38b34600b092396f721b0136279947a] times and only";
        "[scanner-refusal:09ef62c73fdb16b5131522e01174b8f6] second generated-lock marker";
        "[scanner-refusal:fd2664a19380f998eb12b41499b0ca82] generated-lock marker and";
        "[scanner-refusal:a4942423e5944e763f1ac78611e93bd1] names the module";
        "[scanner-refusal:25fdee0fd0c8695a009a7ccdde8a8076] generated-lock marker and";
        "[scanner-refusal:176646e4df25881fadbc9bcd82136c75] Generated.init";
        "[scanner-refusal:22c4d07ebbb0433c6b849bf356be019f] Generated.init";
        "[scanner-refusal:e62fde2478073ff697b31a8548df54f1] dune-file glob matches";
        "[scanner-refusal:90a5bf8d781c4ed0a5407b52bd9fa3cd] arguments the rule's";
        "[scanner-refusal:a02da6b10b325cc5503fe4adf61da286] reaches the scanner-defect";
        "[scanner-refusal:01eec44b44e6e6a758bc7e95ef545443] Generated.init";
      ] );
    ( "stanza_order.ml",
      [
        "[scanner-refusal:a38642dc43749b00aa10a80a44e3de14] the members of";
        "[scanner-refusal:16b77db4cbd5ac82d6ebc373211395fa] the marker comment";
        "[scanner-refusal:122f68556796d56d187e83a664c02f1c] carries the marker";
        "[scanner-refusal:0e5583936dd4737cc5116ef5cb345541] directory's other machinery";
        "[scanner-refusal:0fccc7482c9a39a1e282bd1fbb950fae] the file whichever";
        "[scanner-refusal:486695c03657d82f7e086a00621b636c] file whose tests";
        "[scanner-refusal:61d738fcc3d650e8f79cf987b0379b72] test appended at";
        "[scanner-refusal:1e198bd028e8a39c5c3977cc813d9834] place is accepted";
        "[scanner-refusal:3d1333cd18c39cb238aab1d16996ce50] executable its variant";
        "[scanner-refusal:8dd0857386f3de0a537872e7aa705a1c] under another test's";
        "[scanner-refusal:3a563b87be6f7e65724b94db7fedf3a8] stanzas above the";
        "[scanner-refusal:53db316fd2aa6169c9425879ad300d0c] marker that names";
        "[scanner-refusal:0badf3981a0bc3b5bec0e55af1399e02] aggregate alias whose";
        "[scanner-refusal:c554043991ab4078c3fcc6d153cad7e5] aggregate alias listing";
        "[scanner-refusal:dd4065465913518a0ea00aab99fccf7b] without the marker";
        "[scanner-refusal:d3dfdb8fb0682b0cfbabc46d11b2263f] carry the marker";
        "[scanner-refusal:f9cc533c8648cd034b0c007802ee79ec] carrying the marker";
        "[scanner-refusal:dc68d51c63f61fc901eeda55b3f33b53] at nothing else";
        "[scanner-refusal:ae504ed146c49a8bb38aed572a735ab8] marker was handed";
        "[scanner-refusal:1ce540048aa8e8e8011e29317c3c6002] read holds hundreds";
        "[scanner-refusal:4b1c66381f4f192275a615d13f00d9eb] aggregate alias lists";
      ] );
    ( "vendor_stub_arms_scan.ml",
      [
        "[scanner-refusal:99b67dac359b15961fd222795ccfdc9b] does not read";
        "[scanner-refusal:f214595f668b0bc6bb8db435ac7f68e5] compiles dune's default";
        "[scanner-refusal:8938a0652306caff157652cb8984f7c6] agreement alias runtest";
        "[scanner-refusal:c1457cf67f0481ecff36ea8369db6347] copy_files";
        "[scanner-refusal:7ea0203b4dc3b6f9bb27b060390f8d72] compiles";
        "[scanner-refusal:864cbc7eac4bbb16fefccca605bd57bc] declares";
        "[scanner-refusal:587907379b3593b5bfde494b2785b186] compiles a copy";
        "[scanner-refusal:83773abfec2c1bc90440945fa222600d] compiles a copy";
        "[scanner-refusal:26f527c022e2a36b727d2635a64c83e2] compiles a copy";
        "[scanner-refusal:3f49f1a86b0b50ae1cc45ae30b8d6bcd] preprocesses";
        "[scanner-refusal:b1fb94e1e4593366f815fcc28f356790] the availability-only exception";
        "[scanner-refusal:dee2d648128c5e060fd019f8fb9194cc] Config_key_scan.sources_among";
        "[scanner-refusal:84df002d61c09f8e4c713e4286fdc7ac] Config_key_scan.select_target_of_arm";
        "[scanner-refusal:e62fde2478073ff697b31a8548df54f1] dune-file glob matches";
        "[scanner-refusal:513451a79c3ecc3b1fb18f6fee5ee404] arrayjit/vendor_stubs/<package>/dune";
        "[scanner-refusal:b079d32e59c7b93562738e4061682b90] arrayjit/vendor_stubs/";
        "[scanner-refusal:ec6cead6a475041460ddead2715a32a1] copies it into";
        "[scanner-refusal:b420bc79cbc7a037b600c9d47c9b2525] arrayjit/vendor_stubs/";
        "[scanner-refusal:047374463fd606fe95cf257ebf928364] arrayjit/vendor_stubs/";
        "[scanner-refusal:518629fae55a4ef4b29ab72b3a71d8ce] each module's agreement";
        "[scanner-refusal:f4e3a8ac36957f5a6f54cd735b3c2092] package stand-in declared";
        "[scanner-refusal:6cd3c184704a37f8cb97e98efd6a0a87] beside unrelated sources";
        "[scanner-refusal:90b26d38400dc5a37ed715269ba3cfbe] every vendor-gated select";
        "[scanner-refusal:4cc754a0d44bcdcb048630ccd8e65a5b] vendor-gated arm's stand-in";
        "[scanner-refusal:2c9effa9c434f42e299fea8a5bb96b20] Config_key_scan.select_target_of_arm";
        "[scanner-refusal:1c9ecdb5107c44b922f23d89fb734f94] checked-in ml source";
      ] );
    ( "config_statement_scan.ml",
      [
        "[scanner-refusal:3cab7f51049e66aa35ffa2da5522c89b] does not parse";
        "[scanner-refusal:d565a2d53006c63e583116dbd6d5b497] config-statement reader raised";
        "[scanner-refusal:48f87b9b70b27697be53109fd030b719] test/ or arrayjit/test/";
        "[scanner-refusal:d66c7a38db4797e3caf675c52e78a494] Utils.with_stated_config";
        "[scanner-refusal:fdd6da24ea54afb2db61f480ecd14f1a] the exemption of";
        "[scanner-refusal:248447ba15a28a39a4a4e77731d5d900] Utils.config_file_args";
        "[scanner-refusal:699e50f145d962a2a8fedc6970a58502] read include arrayjit/test/";
      ] );
    ( "gc_witness_scan.ml",
      [
        "[scanner-refusal:bb492e73ce6ee9658023c7166722ae26] does not parse";
        "[scanner-refusal:a141110d8dafdb095f38e629e190938e] GC-read reader raised";
        "[scanner-refusal:cfb5bf5a45572766fd5b7e992f5def00] tree matches nothing";
        "[scanner-refusal:cc1c10dcb5499a78d5036fdb6d119347] test/support/gc_witness.ml";
        "[scanner-refusal:ca35f91c4bd8f24f92904bf594065571] test/support/gc_witness.ml";
        "[scanner-refusal:9f04461988176709dae4ee0369affcf2] test/support/gc_witness.ml";
        "[scanner-refusal:36666810510239397bacb5a2cf2ba433] test/support/gc_witness.ml";
        "[scanner-refusal:e5296137466c1af2ac14a5d6fa9717fb] still forces collections";
      ] );
    ( "cache_dir_ignores.ml",
      [
        "[scanner-refusal:e2e9d19ffe2ad7688396847bd1ee643a] the repository-root gitignore";
        "[scanner-refusal:0f307832357ee7ca58b9798a86530dac] arguments the rule's";
        "[scanner-refusal:88e191145bdba435fd0fb303d486d684] gitignore no longer";
        "[scanner-refusal:3076fb53f1ae34b427f82aaa89404cb3] root gitignore pattern";
        "[scanner-refusal:e2fdd92481a66b85f6f54b5c68513edd] sources this check";
        "[scanner-refusal:af5123353725f010d363e6b3758bfef2] cache-directory reader raised";
        "[scanner-refusal:cd6f72db4a56b656b92efd2c4a07945f] the cache directory";
        "[scanner-refusal:a8ea78ba60f371d22646868a36871617] argument";
        "[scanner-refusal:4168ef8491021497635783bb1f7e4232] no source reads";
        "[scanner-refusal:298f6bafdca761627eb0f4dfecddf6d3] the built-in default";
        "[scanner-refusal:50ad95ff66f64313a7ce58dadb8aeda9] the cache directory";
      ] );
    ( "census_total_ratchet.ml",
      [
        "[scanner-refusal:6d5b4fe654d3592b327d32a16ca1fa2a] checked-in file this";
        "[scanner-refusal:571968c7f074e29e60dddf8d93d23f34] number the allow-list";
        "[scanner-refusal:dd9968e0cfe551a1ba6ee712eef44de8] Verdict's FAILED teardown";
        "[scanner-refusal:f55e77a26cf5f2dde03b8170ef403db6] every allow-list entry";
        "[scanner-refusal:1e1b0223d539c60bc27afd05f8f10482] every allow-list entry";
        "[scanner-refusal:fbdcd3ff3601bf6c0f3343712bb10500] synthetic golden line";
        "[scanner-refusal:9e6321d212778b4b229a787856591c50] number the whole";
        "[scanner-refusal:b7bc91dc2932fa2c119166434ffa3815] match spanning only";
        "[scanner-refusal:9c6e0e6756b498a6378e94ef25883e4c] an allowed citation";
        "[scanner-refusal:f2e0b112265b43aa87718095fdd6dd94] another golden does";
        "[scanner-refusal:b9df30c8c746f670678c71fa8a13e42a] total followed by";
        "[scanner-refusal:b018f905e59688b7aa91c2ca39f76c38] digit run continuing";
        "[scanner-refusal:ad85758e7d6a2b46d81c3d7e6b921d71] outside the boundary";
        "[scanner-refusal:a0e95a750efaba5e8577d41dae60ceaf] one-digit teardown total";
        "[scanner-refusal:4166c0bca69640c175777244011fc4e2] that allows nothing";
        "[scanner-refusal:5aac880a61d2880852d33a0856daa83f] behind a member's";
        "[scanner-refusal:396ecaa5128a1569c6279921019721c1] directory the action";
        "[scanner-refusal:c5c4846ab3c1d492a14be62c4ecbdc6c] allow constructor suffixes";
        "[scanner-refusal:69eae464e65486966fad9baf77c30d51] provenance constructor allowances";
        "[scanner-refusal:f9313f61cc480e2fb8d0b3cd6b1ae843] provenance constructor allowances";
        "[scanner-refusal:ac43499e46e4fb95bbea30412af203b6] counts remain review-visible";
        "[scanner-refusal:84d226496601429f470c41aba20a39c2] adopted-count allowance cannot";
        "[scanner-refusal:992a8057c2edff63b85a6a203b6b5701] harness census allows";
        "[scanner-refusal:fef03fd6e82a5c2c2c628b5a571c8299] census refuses standalone";
        "[scanner-refusal:531d64a37eb2ccc8b4c3fee2c5a2fac9] quoted-source entry spans";
        "[scanner-refusal:22c73fdc7f08c48d003aabefe9f59d17] checked in contributes";
      ] );
    ( "codegen_text_inventory.ml",
      [
        "[scanner-refusal:78fef1ba5a8812597666a8c7a41cf736] the eligible inventory";
        "[scanner-refusal:bbee5d2953722bf625614fa7741f472a] the required inventory";
        "[scanner-refusal:6428d3507ac51411ea25280739f681fd] HIP batch estimate";
        "[scanner-refusal:83dfdd0395c402128515f27b2f3b1b1c] the exclusion for";
        "[scanner-refusal:24dcaa782a3b33780918bb1933206e78] the compiler entry";
        "[scanner-refusal:3f0b314f29005911be716581fa1a4f10] the scan raised";
        "[scanner-refusal:cf8b947a00d5153ebc294985137e0a8a] per-test alias gh-ocannl-1533";
        "[scanner-refusal:8cd31f89229d140f44ef65da81a353ed] is re-run by";
        "[scanner-refusal:564de56163a842792f2175c0c12d75d5] Utils.build_files_dir";
        "[scanner-refusal:d4306fe9c227b85db951de0b64f85ad8] Test_utils.Generated.main_logic";
        "[scanner-refusal:6eb14f027a4be6648748900004341e36] the after-marker exemption";
        "[scanner-refusal:bdce0b6c9330d52290dc2dce205f1b0b] cover every eligible";
        "[scanner-refusal:b1a8055bff15f499f2f143975afca3d7] reaches the stale-exclusion";
        "[scanner-refusal:ae3ea5e30f52522fe3ba79794609f56f] reaches the stale-entry";
        "[scanner-refusal:2c67621c20b4be85e0176da0370f83cb] reaches the scanner-defect";
        "[scanner-refusal:0e0975906df73fda3897bc10ee9904c0] refuses it exiting";
        "[scanner-refusal:88b9820837c81d0eb1237428429e025c] members' interfaces refuses";
        "[scanner-refusal:c79e4912bfeafc46c28c350ff68cf352] build_files/";
        "[scanner-refusal:b3c3d570106d81346b28ef7e49f6c458] after-marker refusal unless";
        "[scanner-refusal:82377b15fd7aac93b28c1deb390f5f55] reaches the stale-exemption";
        "[scanner-refusal:9009ac3e9d8c7293fc8dcced2a1d359c] reaches the unplaced-member";
        "[scanner-refusal:73dda4751ce6710ef3af65fa9d15b66a] unselected-route refusal through";
        "[scanner-refusal:c2effa4a6a8acfb3b5df5bcf40064749] targets combine applicable";
        "[scanner-refusal:123bedafe19167d3536ddb2ca1115d21] cannot say whether";
        "[scanner-refusal:532d3ed986dd6559fec87d780769ee1f] declares module";
        "[scanner-refusal:a59159ac0fd0f431da9519f8016e6762] every scanned root";
        "[scanner-refusal:c995ea0a058a228bf654449befcccb4b] meets its source-site";
        "[scanner-refusal:1b40c8a4f6d202c5b6563c1565f86060] tests building their";
        "[scanner-refusal:9daca1e74abb76588d6d48344619233f] a backend's compiler";
        "[scanner-refusal:be259785a66474faf20368ccde6692c0] support helpers pinning";
        "[scanner-refusal:6e6bfc66aed4b3a13a8be04373cff907] every compiler entry";
        "[scanner-refusal:c278c7bc3f7af4c303d597336e94a1cc] configuration reader writes";
        "[scanner-refusal:d246f03ca5c483db40b94796658cf426] every source handed";
        "[scanner-refusal:87e46487d17e609858f4af46f001e1c3] backend whose build";
        "[scanner-refusal:73a439e337c34309cf51b52176da80ad] alias that re-runs";
        "[scanner-refusal:8879574435d6aa60f13f262f594c3d45] source handed over";
        "[scanner-refusal:ebeeadb92d10e201894cb4b9fb9021bb] every exclusion still";
        "[scanner-refusal:545694a8ea6e371db823bcc7d2639f85] scanned library interfaces";
        "[scanner-refusal:327f5dea331c9591f8d59a6da296202d] scanned library declares";
        "[scanner-refusal:80b2a504ebea9c64f6fcc3c5da12164e] generated text behind";
        "[scanner-refusal:7b848f96a84cc15bb6e1027465d08502] Test_utils.Generated";
        "[scanner-refusal:968edb53fac5a4cfadb338c2a0873278] Test_utils.Generated.main_logic";
        "[scanner-refusal:65ac789025a090533ae12a66e44fab25] every after-marker exemption";
        "[scanner-refusal:ede9d4d135607a80167d420fbbfbbd57] Ll_test";
        "[scanner-refusal:feb6a35da6fc5be9e0f1f066c82d6160] Ll_builders";
        "[scanner-refusal:89659ed1bd4316a75b179801ae387bb5] Winner_fixture's";
      ] );
    ( "config_dep_completeness.ml",
      [
        "[scanner-refusal:2ade05f14bd972f85a1792722f8a0ab6] arguments the rule's";
        "[scanner-refusal:3e9cd231a7db9663f3ceba5ab3891105] produces the ambient";
        "[scanner-refusal:86529712a3b8e4b7a66a3adcb8134331] the ambient environment";
        "[scanner-refusal:6c29eb9b1db92bba229707a7cd8e1b5b] Dune_stanza_scan.action_heads";
        "[scanner-refusal:cebf445823df462f70b459a86b20e3d3] Dune_stanza_scan";
        "[scanner-refusal:5e382153a89be91293964b5417659f1a] build-and-test.md#dune-mechanics";
        "[scanner-refusal:6d151bed80a45201e61d096c2c38d210] which has no";
        "[scanner-refusal:5509acc7fac8ff0412e3631b49c06dc7] change directory without";
        "[scanner-refusal:cb948484fece727200c2dc74529ab671] the ambient environment";
        "[scanner-refusal:e11f1dd0ebe8035f6b6bb919c904e611] this check cannot";
        "[scanner-refusal:e20c1b4ad3881b8147939aef02357ab2] classification for and";
        "[scanner-refusal:332abda7feebdaa6c1c10670865ea5c6] does not declare";
        "[scanner-refusal:c35e7bf895b2bfc334caa198f2db4c83] raw text shows";
        "[scanner-refusal:a83ec7419ab530468bb3d4143c8c8576] raw text shows";
        "[scanner-refusal:28c35655d1568c3c11bdc1893d8847fa] raw text runs";
        "[scanner-refusal:929a18d24ebaad373970a360298e3b88] raw text runs";
        "[scanner-refusal:328553bf93b184a97ab7d44d97c369a4] exempted sites that";
        "[scanner-refusal:c4c555263d62473311fe0cfc7f769958] stanza inline-test library";
        "[scanner-refusal:bdc9f2c15b8f17359eec6d359dfefbdd] that floor applies";
      ] );
    ( "config_usage_scan.ml",
      [
        "[scanner-refusal:f748463375c6c3bb8ff68aaf554ccf03] counted config mentions";
        "[scanner-refusal:3b2d3f635929bbf3953b407c49ee91fd] Builtins_cc";
        "[scanner-refusal:7f7db667dc63d73faefa570451cf5f2b] ambiguous command-line value";
        "[scanner-refusal:7a3d690e9ef5541a1c754952c6cf6877] ambiguous bare config";
        "[scanner-refusal:200477f5fecf3e08ff5055af77a00298] prefix-free config flag";
        "[scanner-refusal:ad6eb9c910ffa6587a73299cb369cb15] \
         docs/agent-notes/build-and-test.md#scan-corpora-goldens-and-floors";
        "[scanner-refusal:1d10abd79d54b54414b3bf881e90620f] \
         docs/agent-notes/build-and-test.md#scan-corpora-goldens-and-floors";
        "[scanner-refusal:8acb2b9087cf4a7e4112382c5e058e1a] Utils.known_config_keys";
        "[scanner-refusal:085d7b5a19f012e46728008a42120c74] non-config environment exemptions";
        "[scanner-refusal:3252eebc4149f3005426dbd4385702c1] non-config assignment exemptions";
        "[scanner-refusal:1b21aa78af1c8c8d33b55c395590e92a] spelling unregistered keys";
        "[scanner-refusal:e592c4ffcb391edf546553bd50accc7a] runtime value separator";
        "[scanner-refusal:cb1b0916f3d396c6fb100a6cf2ba4a2e] scanner command-line prefix";
        "[scanner-refusal:319310c8c133f76c009e4954fa7fc7de] runtime-generated command-line \
         spellings";
        "[scanner-refusal:1a89fa77e81d61087dc6e598948877ea] runtime-rejected mixed command-line";
        "[scanner-refusal:0e70daaacc716dc0bae79f3c55148dff] alternate separator recovers";
        "[scanner-refusal:ebb0dca085588641133d48ef9f80a504] undeclared alternate separator";
        "[scanner-refusal:41ac9c9af9469dc36eb0fdc3623c419f] unknown cross-style command-line";
        "[scanner-refusal:1eca4460778f143859eea42da85bbd72] one-word documentation assignments";
        "[scanner-refusal:7e32dcb7afdd8103e02384cf64776cae] assignment remains classifiable";
        "[scanner-refusal:f39c89d1f700c14082d119fea8a9fb93] tracked historical one-word";
        "[scanner-refusal:6bb17d62436571b3e716c4bba5c0f746] promised non-OCANNL assignment";
        "[scanner-refusal:b8ac42648b6918233995586551724f6f] documented OCANNL assignment";
        "[scanner-refusal:735bd758b11340c1c2fe69f71e3c2cc5] assignment count drift";
        "[scanner-refusal:94c80c1e5ccb09d9d100419e38faa03b] removal remedy when";
        "[scanner-refusal:60932c89821f23ba9cddb3464b18ffbe] reviewed replacement instead";
        "[scanner-refusal:84bf7dbb3e9182b1eced5e76a271e5f6] config-usage direct refusal";
        "[scanner-refusal:03f2822e76feddd4b79a462013315b06] config-usage refusal is";
        "[scanner-refusal:ed12f66c93e310aaa2996f755dd93a06] config-usage refusal control";
        "[scanner-refusal:abf34118c8986fba00fd6d04dbec92e2] source inventory supplies";
        "[scanner-refusal:98de32226b019b81ff6e88bf4e44c14b] supplies OCaml implementation";
        "[scanner-refusal:05a2eb509694607c615f131c6949c1f1] inventory supplies Markdown";
        "[scanner-refusal:df5189a065c11f8226a214b6573c426b] inventory supplies workflow";
        "[scanner-refusal:a51b0bb0846baca8ae1abb956920d3a6] reaches Dune actions";
        "[scanner-refusal:13b4153d0202d6c0c37807c3b9149ec9] ocannl_config";
        "[scanner-refusal:e457cc60db51565a141d184de6993035] ocannl_config.reference";
        "[scanner-refusal:e325a76c47f652bc02203717ce08c44a] checked-in debug config";
      ] );
    ( "dead_export_scan.ml",
      [
        "[scanner-refusal:0379aebd6902dcef1b8a879a586303c6] mli-less implementation modules";
        "[scanner-refusal:b10ed46107421e53b9d08523e194ffa2] dead-export census does";
        "[scanner-refusal:20f449385ebca8a7a25b2d306c8e0f44] scan cannot vouch";
        "[scanner-refusal:ccd19577d4db5de19151a2932010e536] export census raised";
        "[scanner-refusal:a2a944c677a0439ea6a8b98ef3decc70] its mention counts";
        "[scanner-refusal:9dfdb9e8a6d23c9cf5340fe76dc0f63d] type census raised";
        "[scanner-refusal:085de8c521f251df70b65fc6454dfdad] OCaml source does";
        "[scanner-refusal:0c431826a46525bb3cd9cd36a4c9e833] reference census raised";
        "[scanner-refusal:e8c85fd2a3138c8fe3283ab67595826c] zero-reference implicit export";
        "[scanner-refusal:8e094223fc3e55912db0c3f53c9fcaca] named dead-export exemptions";
        "[scanner-refusal:e32dfff95d976581bde669214a7e6415] dead-export exemption remains";
        "[scanner-refusal:648c21e86192530edfa91eb57a98cf3c] every type mentioned";
        "[scanner-refusal:1196acbcd8267dcdc62af2676f1a223e] named unmentioned-type exemptions";
        "[scanner-refusal:2bb5b26c546587c6cb8afce38d28f8b4] unmentioned-type exemption remains";
      ] );
    ( "limits_field_classes.ml",
      [
        "[scanner-refusal:6dd30560f9e032ec9afc45a6fec9ce97] every field of";
        "[scanner-refusal:280bb27b49e792a7b0d751509c083f70] every toggle here";
        "[scanner-refusal:c572cbe3fb10a48b7182cf818e45bb75] score's capacities residency";
        "[scanner-refusal:99e4f592f87fc2294a969d3f7910070c] the Reporting fields";
        "[scanner-refusal:25c581607071ee5d7f1c6e69d9f2dfe0] Reporting field leaves";
        "[scanner-refusal:19c207848ef78f880b97d63600612985] Reporting fields absent";
        "[scanner-refusal:0a9177a9cf5f80388c5ecb335859d27d] the binding pattern";
        "[scanner-refusal:129a692bf5913667539ef35c0825728a] reaching the accessor";
        "[scanner-refusal:321a18c08b264e43b8d54ef69e4786bf] the definition reaching";
        "[scanner-refusal:b4baf80e206eea100c4257a2cb0cc11d] fallback every search-less";
        "[scanner-refusal:5de8818597d1f7b0e33446b61b0cf363] backend_intf.ml";
        "[scanner-refusal:8cb8dd2d540fbb23c426e1e40d98a1bd] untimed_limits";
        "[scanner-refusal:c5b36a6c7a3ca9df76f09052570d546e] reporting_limits";
        "[scanner-refusal:393f73abc235553841f56b85cc181a23] hardware_limits";
        "[scanner-refusal:8f9176198240323dbd1fb605e9c46ba6] backend_intf.ml";
        "[scanner-refusal:afe5c720e31bee3c4413dcb920198bb4] backend_intf.ml";
        "[scanner-refusal:ef99980e8fef8e93f88e9ada52db5875] autotune.ml";
        "[scanner-refusal:eb09835831cfa5c8bfec24cfddb4cb71] autotune.ml";
        "[scanner-refusal:190f5d5f9cf7b1a188c423cc9c9159e6] arrayjit/lib/autotune.ml";
        "[scanner-refusal:887690bba79db7186f007c0ee3bc4d55] autotune.ml";
        "[scanner-refusal:0fccc4731b902ce5dbc0c1a0ff5289f7] autotune.ml";
        "[scanner-refusal:cabfbf0ca35a6bd413475c234f5cbd49] mma_peak_leg";
        "[scanner-refusal:00ed6f66ef1266b0be031d37ef0bbd03] autotune search ranking";
        "[scanner-refusal:02907ae08467c5deca8ec391bc107c00] model_default";
        "[scanner-refusal:fe049afe671ea46b4b4e1a97110057e0] autotune.ml";
        "[scanner-refusal:83edca568007fee6d08310746a95bf8c] search-less branches test";
        "[scanner-refusal:59dec6ead5ca6a8266b3739f62424e79] seeded transform reaches";
        "[scanner-refusal:fca7b1c5180357d7ba51bda9f129d30a] fallback every search-less";
        "[scanner-refusal:ca0e056f71e7b3b2d315db155984df89] definition tune reaches";
        "[scanner-refusal:34c994839232ad03792cf504a0770b5a] definition tune reaches";
        "[scanner-refusal:2c95f25304a68db188b11a98b9b6df51] Autotune.tune";
      ] );
    ( "digest_completeness.ml",
      [
        "[scanner-refusal:7cab0a5763f4811f331a4aab327c6784] which the codegen-stage";
        "[scanner-refusal:fc3ef3788212f2d1f5a9d9a46be3ca38] keys classified more";
        "[scanner-refusal:dc9995a5d5b80370dc90971d3fd019df] no cache-identity classification";
        "[scanner-refusal:a385ae718c8af3ab39a45701b75fda4d] Utils.known_config_keys";
        "[scanner-refusal:14b02d2180d3814d6746acd827aecbc3] classifications claim cache-key";
        "[scanner-refusal:8109160f59717560ff5e6f7251a6a2b2] cache-key components no";
        "[scanner-refusal:1825a277ba0eda3313fbe079e465c15d] but classified code-borne";
        "[scanner-refusal:3a04e06161978f8255007c739f6806a5] Autotune.tune's";
        "[scanner-refusal:d011313dec8b9e6e53f58ce97ef08a66] Autotune.tune's";
        "[scanner-refusal:12aa84c890c2e9d7e1f3bdca30397d4e] placement-shaping or untimed-model";
        "[scanner-refusal:55743888dd2a4a2234ee1d3f031ea47e] meets its source-count";
        "[scanner-refusal:a9c0305e1a7d7817b8ba147ce52e8f90] classified exactly once";
        "[scanner-refusal:5e2800ca22fd1c7629755f638e39e542] is classified code-borne";
      ] );
    ( "env_var_deps.ml",
      [
        "[scanner-refusal:57263e03b8b032365fbd586bbd2705a3] stderr from Verdict-linked";
        "[scanner-refusal:2b65edf20e36d983c1a016b72c2aadd1] plugged-in check receives";
        "[scanner-refusal:35bab4283cd8b87d51ca3476cb0e971e] source_tree";
        "[scanner-refusal:51b4c42723bc3fa6bfc678d091151ed9] source_tree";
        "[scanner-refusal:3182e84964718fd8189a6dd0d93d14b1] diagnostic absent from";
        "[scanner-refusal:8daa7afe8b321ab7932dac6060588501] same diagnostic fragment";
        "[scanner-refusal:08ac167f3e68d870e2cfb7f7c85accc3] control marker occurrence";
        "[scanner-refusal:e68d9277c8a3a5c061164578baf2f632] the population equality";
        "[scanner-refusal:87659b49df04eee3fc19350b0be6629a] repo-relative scanner paths";
        "[scanner-refusal:0e5ddf66a7ff3f9265d5d35a729daec5] by-name environment reads";
        "[scanner-refusal:d7d902ac3600e16a27e9a62229789cb2] tracing-gate walk raised";
        "[scanner-refusal:8dbf084aae3ed7d3e4ce33fb0d06e710] the credential deny-list";
        "[scanner-refusal:490ac5e3405d328c2953c38df4f0974b] arguments the rule's";
        "[scanner-refusal:2dc2479687a000324b3f828912a7f4a8] search derivation lost";
        "[scanner-refusal:a81eacf691e2faeada3ee2b79d5fd7a6] support values whose";
        "[scanner-refusal:50ad2bbac7aab894b82d2525afd2aaf8] that stanza's marker";
        "[scanner-refusal:fa823e2140df2fbae2d15be1794a146b] can select cc";
        "[scanner-refusal:9de3b3fd1cd96a8d87e76bfa31534244] comment that does";
        "[scanner-refusal:e96ac262e859176f6f0ec05d6f2f30cb] the wrong stanza";
        "[scanner-refusal:0a10ab654cc394b5f797d88ea889d086] inside the parentheses";
        "[scanner-refusal:be4bb5e070f7c1653c2f7236cb2e6a18] times and only";
        "[scanner-refusal:034280d144ee4338406ff5f6a1cf287d] comment that does";
        "[scanner-refusal:30c88c1631600d7969a9aefd4de74162] carries a backend";
        "[scanner-refusal:60a17b773827a71dd5c724af51013878] its action through";
        "[scanner-refusal:bd8470207f6c45ce4020275b78402293] fixes its backend";
        "[scanner-refusal:78e6a1f5c4350e90995d5b6e874d7e3b] pins the backend";
        "[scanner-refusal:f2633aa4ebd6f2c33fd21e4d69cc9786] one backend marker";
        "[scanner-refusal:2803edcc7de84ea6d928343b9d8c8ad2] env_var";
        "[scanner-refusal:533fc9999ad0841992e0c1734d9c1f5d] env_var";
        "[scanner-refusal:7336dd5a712b0c3a578b64d7defb1654] between stanzas declares";
        "[scanner-refusal:1aaab6e3cd8f4615e1cd1d0321e309a1] times and only";
        "[scanner-refusal:f11c09a8318b3a7863444256973f57d4] running an executable";
        "[scanner-refusal:034280d144ee4338406ff5f6a1cf287d] comment that does";
        "[scanner-refusal:6d4652d347d64362ce0c9b3c1498abb9] carries a backend-default";
        "[scanner-refusal:eeabf0209c60b38c2cf243f1fcbaac4a] backend-default marker that";
        "[scanner-refusal:f75580a3f9d63e77e8203a3df984bb9c] times and only";
        "[scanner-refusal:942031ff44cb018ae2d7eee70ed6327a] the backend default";
        "[scanner-refusal:95919586e2a2b25eaaf13ca8093ab584] repository-wide scan rule";
        "[scanner-refusal:be41f253ce4b808b6b4ec587c304ec08] repository-wide scan rule";
        "[scanner-refusal:b040632469771b25abb3469878413be3] the repository scanner";
        "[scanner-refusal:b7991c01507b15cea041c6066f07038b] names the training";
        "[scanner-refusal:2a4ae5808191d606dc5b7b74dd4ef93e] names the lock";
        "[scanner-refusal:bbadfd8513559355fedbc54dc3dc678a] serializes its actions";
        "[scanner-refusal:eb5d5419b3bd18240469743cbabc58f2] has actions on";
        "[scanner-refusal:a06dccc27afe2153a7700609691d3c8d] attaches a rule";
        "[scanner-refusal:7f157a5478f43aa3d2da34c9975976a9] inventories the repository";
        "[scanner-refusal:91751d43d8939e1da724e6f0bcbe8f09] inventories the repository";
        "[scanner-refusal:f16784d0b1f52de5e30378cb771ada5c] alias does not";
        "[scanner-refusal:4012b24c7623acd977522e8f2eacf2bb] attaches a rule";
        "[scanner-refusal:ba5d90fc4bcec944ebd169329c706968] goldens on the";
        "[scanner-refusal:c1a398b587b5de3d7d02145dc45274a6] alias <suite>-<name> that";
        "[scanner-refusal:941a9fb77d867c7b1576a0f19ee5db3b] declares";
        "[scanner-refusal:7aeca22f89e6555c3b97967f8bbdaba4] credential variable on";
        "[scanner-refusal:92d872e8d8f1d6a3de97bf3499a7170d] env_var";
        "[scanner-refusal:0deef0cb94c45615f32ee84ac2e76f8e] declares tracing gates";
        "[scanner-refusal:02e183a4105132ec092206c1d35d6247] reads the tracing";
        "[scanner-refusal:2df546d04b63387a24409b3c4c6a9611] declares the tracing";
        "[scanner-refusal:a53c6b4b54cc85d2f251e80dd90c855c] credential variable on";
        "[scanner-refusal:e55f77a62871fb685d23199e95c8f173] the environment variable";
        "[scanner-refusal:b9c87589aecbe6de4db58fc4671f7491] include_subdirs";
        "[scanner-refusal:eac17cf643d980bb15c3113087c863c4] names the module";
        "[scanner-refusal:e47faeb8aa90e38e3dd7b698f6ac8998] carries more than";
        "[scanner-refusal:a45f7f9b519fd1f5ef3516593179eef6] cc vector file";
        "[scanner-refusal:22289447396ba1e1150aeaeb5faefadc] starts a search";
        "[scanner-refusal:591d750b2330cdac197127abc39ce397] executable's own modules";
        "[scanner-refusal:f30ace350fdf01fcc307ea64e391a7bf] Test_utils.Generated";
        "[scanner-refusal:b0ff4d4e5709299b84685f0fc189cc63] the inline-test runner";
        "[scanner-refusal:5893c6332b1f903470ec51cad4f20e8f] holds inline tests";
        "[scanner-refusal:5f2b09efad6e397fb5bd77b84255d0bf] runs inline tests";
        "[scanner-refusal:677d819fac904c64b7025b4f17c07b5a] its stanza starts";
        "[scanner-refusal:ffa1cdde8b0746fcdb7a1ac0b406107a] the backend default";
        "[scanner-refusal:2d95d96433c72c5c970b888087a59832] the backend default";
        "[scanner-refusal:6c8e61d2480bca6f93d3fd99b945e5ad] the backend default";
        "[scanner-refusal:04f8a4dfd1058a7595117f80bc1fcb0c] the backend default";
        "[scanner-refusal:66742d7bb3766af11b75ab38bf798fdb] the backend default";
        "[scanner-refusal:b0d8f5f604f61eb8ecc4829d93f24a83] the backend-default reader";
        "[scanner-refusal:afd2f461af706b40bf4582f1b0a30b01] the backend-default reader";
        "[scanner-refusal:77d0be0fe2fca9a97da8be7f833d9c6e] is spelled without";
        "[scanner-refusal:25657f61eca857f6fd2f34de66e81eaf] scanner refusal";
        "[scanner-refusal:fe4c53f899eb08a9ae67e7a797a5a841] scanner-refusal exemptions no";
        "[scanner-refusal:05b0b536b88908d4ef277447d7e82ae3] exempt_reads";
        "[scanner-refusal:638620a0745d0a136810c5b0256e0ec0] directories exempted from";
        "[scanner-refusal:7d1923374b5bbf7028b59f2670772e8c] executable either declares";
        "[scanner-refusal:b8641dbf725aa68ffcc15fea6a3eb550] ocannl_config";
        "[scanner-refusal:68816c7803a2eabaa2a2d8799ba00a78] running an executable";
        "[scanner-refusal:a90efecb7feb61485c0886565a0e4ba9] running an executable";
        "[scanner-refusal:09ce44850cdf45c95e308d6c64aed813] the repository's";
        "[scanner-refusal:1bb3be04ec494f4917d1537675671f1d] repository-wide derivation finds";
        "[scanner-refusal:f8d375e9267abafa87ce8eec2d53731d] repository-wide scan rules";
        "[scanner-refusal:f1c1e39705142c7bdc92b3f0ede72582] refusal-control manifest source";
        "[scanner-refusal:1f42d7a2739170c96004439d524c5d96] permanent control-golden corpus";
        "[scanner-refusal:a2ef9338814a08816bb85a6bab6ac71c] statically recoverable scanner";
        "[scanner-refusal:edc9c30307628c25acd76b6959b5a172] record carries declared";
        "[scanner-refusal:b127e705a9263d141f051fd084d920f5] unknown lowercase name";
        "[scanner-refusal:4a289c3610d53e6f94b45fa13f689381] someone else's variable";
        "[scanner-refusal:055900a8a1fbefd29b7c64a46953700f] external command handed";
        "[scanner-refusal:fac924c455e2114b6e6bdebdc44811fd] same stanza declaring";
        "[scanner-refusal:b4a6c6ad5ae1c0d26b7fe78da5c5aa7f] command handed nothing";
        "[scanner-refusal:543a9231648ec3ea9aea18b9d4a7a989] plural-unit declaration yields";
        "[scanner-refusal:879dc483f2a44c38db9d0f5bdacb746c] instrumentation's Dune directory";
        "[scanner-refusal:adc2daf0a8a1fc0ba72f8f863650f4af] instrumentation ownership invalidates";
        "[scanner-refusal:e36e9460538a22f74c5e20a589002e2e] dependency path without";
        "[scanner-refusal:5f90ec9ed44c4e3a43172e41e2ae7503] fabricate instrumentation capability";
        "[scanner-refusal:a6ad129bb27775f8c16abc37c6c39ed6] metal-codegen alias does";
        "[scanner-refusal:d2a4a4c55f9409ff2a808631d0753f27] family stanza passes";
        "[scanner-refusal:d53b323fdb52eefd4e3968dbadf05e65] with instrumentation-capable libraries";
        "[scanner-refusal:da4ac19271accffacfcd9a6bbe210e22] member listed passes";
        "[scanner-refusal:bb441071afd26790b8593e48298b3afb] local declaration remains";
        "[scanner-refusal:823899fe97c22285dae0552d086d404b] passes without resolving";
        "[scanner-refusal:3f9f68401bac53be0cbb38ce715e460d] executable whose RUNNER";
        "[scanner-refusal:ed1dfb1acf49ea77d9b2bdd8ad92d312] alias listed passes";
        "[scanner-refusal:ddd1defde5b6df4dab1d802e19cd88ef] generated aliases listed";
        "[scanner-refusal:46c759bca27349aa081cf5efe33bdef0] same plural stanza";
        "[scanner-refusal:60f9c34c0fee7c424caab53c50c2a878] same one-alias listing";
        "[scanner-refusal:38cc9ab62dfcf93c10a811d1a8f37ecb] inline-test library carrying";
        "[scanner-refusal:97233d85a47856acfecb615dd26a3ebc] generated alias passes";
        "[scanner-refusal:e45a490c152af6ca00b7e83c84599d95] recursive build from";
        "[scanner-refusal:a0fe83d0193d8ac517eb248c21c7f120] another directory's executable";
        "[scanner-refusal:45113608604d50fd82c92657c03134f4] this directory's executable";
        "[scanner-refusal:cf759af799dd89758aa6f0eff3cd03b5] public_name";
        "[scanner-refusal:55ce549b47e5402396164709a1e7564b] top-level family stanza";
        "[scanner-refusal:82254747d84d2b7698c0987019df1914] the directory-wide runtest";
        "[scanner-refusal:0ad19c6a66fdbd030d2b742800df58bc] second unit's public";
        "[scanner-refusal:7735cb3446daca7f4157aa26d3d583da] names the instrumentation";
        "[scanner-refusal:248545b295fe9393f046d06407875db5] family alias defined";
        "[scanner-refusal:65fecdef5d771aadd55c653ca5f36ccd] group whose actions";
        "[scanner-refusal:3200824ccd2f1707a2ea135232ae64ed] per-directory lock and";
        "[scanner-refusal:93882a7d0ee00a277420de73a0971879] refused naming both";
        "[scanner-refusal:61d28abcff988cb53c2d14ed6a3c9b08] same relative lock";
        "[scanner-refusal:9cf07fbca9351e2f1f3a8ec460854813] sibling directories resolve";
        "[scanner-refusal:869d4851cd153a86c278696b8d6b4f89] spelling written in";
        "[scanner-refusal:b4e757c29c3b7fcc02f6b6166b7321e1] compared across directories";
        "[scanner-refusal:84e456282a1ff74968abe67ae2fa4694] executable's public name";
        "[scanner-refusal:be7f67353f6c0f8782c3b867b1023af5] runtest-<name> dune generates";
        "[scanner-refusal:4a8965b97909e604cda0ce07fd679917] a lifecycle member";
        "[scanner-refusal:27dfb3ff299833db9842614f2e2b5384] local executable's name";
        "[scanner-refusal:0880ce931733097b9e44c86e1f0b7065] ANOTHER directory's binary";
        "[scanner-refusal:93ed6fc086891252db10d64e8dbfeca0] alias dune generates";
        "[scanner-refusal:84a1eda5072c0c3e43ecad5c2af6e561] stanza neither derivation";
        "[scanner-refusal:ed4970402e1305b0dfaabed50860c3fe] configuration variable the";
        "[scanner-refusal:c5caa846079a8640d0b93242bba8fd1b] tracing dependency without";
        "[scanner-refusal:144d103d186b613dbedbe258066e8ccc] whose read disappears";
        "[scanner-refusal:04c4a735214f08537c76d2eea48da974] family stanza listing";
        "[scanner-refusal:f48208137f226b2a88a3b74e26606f32] Ir.Numerics";
        "[scanner-refusal:d5993b75f466d86b8b85f936920dab21] Test_utils";
        "[scanner-refusal:07998ddbb5d57aef4a47cef70de23836] numerics policy only";
        "[scanner-refusal:1079839546092d96868e5a453cccbc0b] opted-out member the";
        "[scanner-refusal:fb28546c629f972f641023b13bbadcf4] opted-out member the";
        "[scanner-refusal:88592385a9799d9f98c3be9d93123e2c] member is refused";
        "[scanner-refusal:59a004df0ff72eab5f2475d0ac0a92a3] opt-out declaring anything";
        "[scanner-refusal:971bc72210caf9086923f04657b45008] unconditional family stanza";
        "[scanner-refusal:f153e264ba5496c0a9f14ea067f21c96] conditional member listed";
        "[scanner-refusal:e73d0c1ae2e2a075dd7e0de834f08ab9] executable naming the";
        "[scanner-refusal:e9bf35fc126f302b1c2b4eb4e3f8224d] executable listed through";
        "[scanner-refusal:b8514eb10109dce5b36ef105c1bc36a6] module passes unaggregated";
        "[scanner-refusal:9dc8db9d29efd1b85da8aff105b1aae8] tested module with";
        "[scanner-refusal:775f43c0a7dca22c07a091f603457c7a] another is reported";
        "[scanner-refusal:87104dd3758037b44d4f5c8979a42039] per-module alias listed";
        "[scanner-refusal:d04fa9759a77fcd06752e1d4a912eb06] source_tree";
        "[scanner-refusal:760e7621ae0370f92b10878f9d71ee0f] source_tree";
        "[scanner-refusal:31948a4b787d4a96777593c36d38abf6] repository-wide scan rule";
        "[scanner-refusal:edcdbf38cf4e4cb1afbe781c92620485] repository-wide scan rule";
        "[scanner-refusal:cc3ad53b0e8cc88b373a1a474d63f0f7] repository-wide scan rule";
        "[scanner-refusal:56b7e1396b8287e30b29b436fc729e13] unrelated configuration-reading alias";
        "[scanner-refusal:32ebbcf3860e0aeff7ce6014cae7dc77] their configuration-free exemption";
        "[scanner-refusal:33d6f713bd21678df8dc25c1f94de29e] configuration reader attached";
        "[scanner-refusal:8e23bc2547246ce32e080d7a985cd246] a configuration-free exemption";
        "[scanner-refusal:bd0f7175b0bbdb8fff16e6a342879ca3] unresolved alias dependencies";
        "[scanner-refusal:f8f322e105a5e0f26e1dc4dfb4a25c7f] missing scoped exemption";
        "[scanner-refusal:ca87c0e84f8b051f232796e69330144f] renamed direct-utils universe";
        "[scanner-refusal:eb2b142cf7df4f4e59facf55554fb04e] universe-dependent compiler census";
        "[scanner-refusal:474c2c17b390e6a30e2a618348b20390] another directory's same-named";
        "[scanner-refusal:64dfdc824dcf91354dbd2abd219db619] universe rule running";
        "[scanner-refusal:06fb7574357e1a85e9d7da69c19351b3] gate cannot provide";
        "[scanner-refusal:3bacab0b9c1d831e547e7ededd7085e2] generated-alias collision exemption";
        "[scanner-refusal:6d2cc639b272c289c3a79c46561b671f] explicit self-run resolve";
        "[scanner-refusal:314daa1e6e83eb37702ce6cf4f24cd44] ordinary direct-utils test";
        "[scanner-refusal:f8fc95de59a65622241eac16f5f1ecf4] effective gate declaration";
        "[scanner-refusal:abc806b50e485449683b10cc952fde9d] preprocessing-only universe \
         dependencies";
        "[scanner-refusal:0d805a9703eb5f183046f05353ed6587] expansions and subtracted";
        "[scanner-refusal:9fe5330074251fed6b9e5e08a0d663bd] universe users alone";
        "[scanner-refusal:1980131e56fcd4b8e4810ba218228fe7] whose rule depends";
        "[scanner-refusal:a6af1a4eabd80e213ebd285160652900] argument stays ungated";
        "[scanner-refusal:4bff8c4ce57f4125cd336ce630eaf5bb] a program starting";
        "[scanner-refusal:fe8028c677482d15c38a4bfbfcbcf772] Autotune.Make";
        "[scanner-refusal:b458bf994a8764fcc21c183cb3f0786c] Autotune.Make";
        "[scanner-refusal:14853da159c407bfbb2fb7e5a6597d9f] instance until shadowed";
        "[scanner-refusal:3add57bb89dce424f18b91de79486f58] initialization pin must";
        "[scanner-refusal:1941f1fd1c6d01e76c592bdab3261140] search item invalidates";
        "[scanner-refusal:4aeee97cadd279f1e1d00bd64a116077] search item invalidates";
        "[scanner-refusal:c3b6672a6f1e65c6ba16b1c987d1a8cb] invalidates an earlier";
        "[scanner-refusal:aa2e14b6af61751440ba04ff7709d6e6] invalidates an initialization";
        "[scanner-refusal:88245fbc788612e3501d835011df08f2] interface-only sibling module";
        "[scanner-refusal:4b7124f3852d2e10d4a830de32bdedde] carrying cc-vector-width independent";
        "[scanner-refusal:974939e6b89b36d455f0e19ff70a42d3] pinning the variable";
        "[scanner-refusal:687a8e82db5772147024b757deca3cd3] program whose initialization";
        "[scanner-refusal:720484df63c40ad6dd2418d4cfa706cc] program's own statements";
        "[scanner-refusal:6454f0cfc83773267ad5cf7695226069] helper printing its";
        "[scanner-refusal:be7654cb2fedf501f96178ab18687738] search started through";
        "[scanner-refusal:7fbb55a505a047a588bcf2bfe6df0f36] program whose only";
        "[scanner-refusal:c6749493265651edae753b4d91b6057f] program that starts";
        "[scanner-refusal:297ca2b13030714ee532288bcdea6931] refused as contradictory";
        "[scanner-refusal:008feff3fd3ac8bf0aa213425ccab973] marker declaring anything";
        "[scanner-refusal:f73bf9597915d82e0263c79001cade55] stanza are refused";
        "[scanner-refusal:46430b0f534e84dda590f0e5cb983438] marker's sentinel outside";
        "[scanner-refusal:d3576c4f9152f9f5a8f7b5bd17bc7e39] no program's modules";
        "[scanner-refusal:435bce2f838f426728b540ba5f6c905f] marker between stanzas";
        "[scanner-refusal:3956653315998753df29070b7b38ae6c] entry point renamed";
        "[scanner-refusal:eb27e7f56b8d3460a37076b391593d9e] configuration table pins";
        "[scanner-refusal:22825750e1b48f737f60249e27047e28] empty string which";
        "[scanner-refusal:d3f1d0e95f110142f78dbbfade3eec72] through another module's";
        "[scanner-refusal:90ead9391fc487dfbe2cd91e0b0f7288] helper's module does";
        "[scanner-refusal:8bda63bc4c91bc8f3c9f625a9c7f7f44] program pinning the";
        "[scanner-refusal:77352d3abfb3f47f0dcc785a2e428b15] an initializer's statement";
        "[scanner-refusal:d65197a8a018021b4c7aa22fcf90c773] namesake module rebinds";
        "[scanner-refusal:593b4e2508d631a9f42d0d6fe061b2e6] program pinning itself";
        "[scanner-refusal:e1e7c11bf8ecfecb20b9031955b16937] helper called after";
        "[scanner-refusal:63423f2793604647b96a315b851f1859] ocannl_cc_vector_bytes=-1";
        "[scanner-refusal:48335fe342f8e4b1cdf081db8783e475] search started through";
        "[scanner-refusal:0876bfabcf2ed9abc14e936d4ebd9018] command-line setting in";
        "[scanner-refusal:25394c54f621666e9eb6b4419cee87fd] closure the initializer";
        "[scanner-refusal:3d6b57503b5fac620375759535912df9] spells again before";
        "[scanner-refusal:585ea2db9ead892ec918b32c984e2d49] is_cpu";
        "[scanner-refusal:c5e986988e4412345d94af936dabc9d7] helper's module which";
        "[scanner-refusal:2555a3d542f36ae8f5d74c20d99d9cde] run's invocations undo";
        "[scanner-refusal:63f4eab503a1525a9de1686cfaca3b41] support value stating";
        "[scanner-refusal:4cf5336639f62b774e8c76bc09d17120] re-running a test)`'s";
        "[scanner-refusal:146f0968d410f75502805f05d1a387b3] the program's marker";
        "[scanner-refusal:a625fef5b0ce99db1e03c615491fa578] undo the initializer's";
        "[scanner-refusal:416d761ba145a451f16c4913c93ec004] re-running the program";
        "[scanner-refusal:194c3da6b4172812b49f3a686fb02e5e] handed an executable";
        "[scanner-refusal:28d540b06acc9c02ad1d408123cc27ae] text-valued pform which";
        "[scanner-refusal:9b37c5d94470adf77d607d70645a584a] interpreter handed only";
        "[scanner-refusal:d29c340cf3960a26e7f234805bcebd18] the declaration checks";
        "[scanner-refusal:d896c784ca87717dc5c6d91befbce0e1] source is refused";
        "[scanner-refusal:fac99db54dbc751b212940875f53c769] captured Verdict run";
        "[scanner-refusal:edc4786c103754e1274b9e12cf08fa3f] nonempty binding undoes";
        "[scanner-refusal:e41ce36dfe9ad79d2112520081833ba9] captured Verdict run";
        "[scanner-refusal:3f8d91b20ff8dd63fdb0fa1e1cd5cc7b] select stanza passes";
        "[scanner-refusal:cab6ce652200dd6b791b352310a35bdc] undeclared tracing gate";
        "[scanner-refusal:4b254ada2beaa28473ff82577bf198ee] undeclared ambient env";
        "[scanner-refusal:1692b7fcf8f2e58a4884d397c7d39c9c] non-default select arm";
        "[scanner-refusal:f1f5e5ad4613c408d71f8a02b4462be9] non-default select arm";
        "[scanner-refusal:4112d0c301944346aa5b2b228db48361] non-default select arm";
        "[scanner-refusal:c32893cb68f676d5b920e664ddd544a1] default-module-set stanza discovers";
        "[scanner-refusal:cc3f3f4d2c2871eaf60a3bbe9f670f42] invalidates an initialization";
      ] );
    ( "ocamlformat_ignore_scan.ml",
      [
        "[scanner-refusal:e1f2fc5fc55abe7d73725c939c81a516] ocamlformat-ignore ends in";
        "[scanner-refusal:d936cdade68aaf6cb4f1a8a0bfc17655] ocamlformat-ignore line contains";
        "[scanner-refusal:cf335754f478997f3146ae0063d7d03e] every ocamlformat-ignore entry";
        "[scanner-refusal:2e170f2ab6d02300e60fc334c88d3481] every ppx-expectation golden";
        "[scanner-refusal:bd075431f78cab212b8ec01cde19088f] complete fixture with";
        "[scanner-refusal:050d763773d2e7b11b18179905cf9981] concatenated append is";
        "[scanner-refusal:f111a87fee55ca52831220a49495432a] its append-corruption diagnostic";
        "[scanner-refusal:d3e7969b1266fa6b311234da304986db] line containing no";
        "[scanner-refusal:4186a2cb3aaa270f0e693d87e91d2e2d] undeclared file present";
      ] );
    ( "command_line_length_scan.ml",
      [
        "[scanner-refusal:372eeaf368300a9e358067e8d495cf72] the repository including";
        "[scanner-refusal:37ae8aab17397f20999234cd8d0f6736] dependency";
        "[scanner-refusal:731bfaa30bb282b9a618b0bfdd29e773] dependency form";
        "[scanner-refusal:087813b8ad2e0e15e1f1cf137190ea8a] target spelled with";
        "[scanner-refusal:d167faac8f0029e909ca0dc0f4b30e41] stanza with no";
        "[scanner-refusal:d2e24c0f204c468788a307dee2026b3a] stanza that names";
        "[scanner-refusal:af863572668f2295ed85b51e4b34c459] puts the contents";
        "[scanner-refusal:1a514655621d234733eb18e735300edb] pform this scan";
        "[scanner-refusal:6cdafd68dfe90265d75399ca116fd389] argument";
        "[scanner-refusal:cf6423e820c211ef1f7669e8f2a5bd80] one this scan";
        "[scanner-refusal:979a75b9cff181cf8c618e3265f341a1] directory this scan";
        "[scanner-refusal:f659c7db4d5a7eeb09bb13eee5dd0d9d] action head";
        "[scanner-refusal:0292958ea115eba550f5b11a616b213f] unreadable dune structure";
        "[scanner-refusal:b2a275ebebff3747d03efc331e31059d] includes stanzas from";
        "[scanner-refusal:9d49d08dd876c701ec8baa44b3e8055f] spawned command line";
        "[scanner-refusal:7b1d721a6b358055eeef13e57e5ba202] source tree reads";
        "[scanner-refusal:69918b599474e445c1e2dce74dc3e841] every spawned command";
        "[scanner-refusal:6eba8bdcf1112c04fdf6c29d4770c94a] bounded spawned command";
        "[scanner-refusal:15437884bd1d780ac790343204eab7e4] every judged";
        "[scanner-refusal:e22a999e7b876b3b14333e3000c76c04] response file default";
        "[scanner-refusal:407d4f902a30caffd73b344ba63b6731] gh-ocannl-1720's shape a";
        "[scanner-refusal:23074ca9ed294d7115c4a6c3d3c6e5eb] named dependency bound";
        "[scanner-refusal:fee1a23a89df95042e6931bab5c95fa5] whose expansion reaches";
        "[scanner-refusal:51a97beb27d22afda197ecf8855fa059] line that reaches";
        "[scanner-refusal:b605b3e6ab959664f78a63dbbbe15d16] cmd.exe's";
        "[scanner-refusal:6438c6083e355a1d9525b4508fc34320] bound is refused";
        "[scanner-refusal:8abbbf622b67401b3c6ed085c0e56eef] no judgment bounds";
        "[scanner-refusal:1156b111ec97fe778b6dbcba4ce8e21c] head on neither";
        "[scanner-refusal:c89bc9f1b5b4e0adddd25e00a9c88da7] scan cannot resolve";
        "[scanner-refusal:d71d7140161f2e19abd0f58123eaea02] an include stanza";
        "[scanner-refusal:f2f50b0ccfbbada97516cea48cbc3f16] file sexplib would";
        "[scanner-refusal:b09b0066f0f41d3b0b1ffc188b84c55f] a dependency mixing";
        "[scanner-refusal:cde1da25360bc3b02f4c015d1c5239cc] unknown dependency form";
        "[scanner-refusal:98522ab6abd0e6c4b64b1252f37154a6] whose field contains";
        "[scanner-refusal:ca1105d4fef574c3c55a3a5ce5af87a8] target reference without";
        "[scanner-refusal:c223f8813d0c15a9a59ea92bcfa4b2e8] test reference without";
        "[scanner-refusal:2c815f3192c897e5f51220a7b538a931] in command arguments";
        "[scanner-refusal:5592d35c9de1404b0763bd99eb0cf0ab] malformed spawning action";
      ] );
    ( "optional_arg_inventory.ml",
      [
        "[scanner-refusal:924efa1e307f7c5a40712dd66311abe1] the inventory nothing";
        "[scanner-refusal:52e64fbae740ca90c4066b7840e926ae] source hand-over reaches";
        "[scanner-refusal:8827ebeb8572d08d2dd427695c19c0f4] lib/ optional arguments";
        "[scanner-refusal:69241f287bacdd3e1e418debbdc13d3d] every unimplemented option";
      ] );
    ( "shell_scripts_parse.ml",
      [
        "[scanner-refusal:5a33b3090d0b22b64b0dcfd7eb75c717] dollar-quoted heredoc delimiter";
        "[scanner-refusal:5a100f6085e18fab300f003899e342de] unterminated outer heredoc";
        "[scanner-refusal:2c41b5db925a6aa0945ff98d5c2c7625] continued unquoted heredoc";
        "[scanner-refusal:261986b232e3c0d320e14e162a45fd02] multiline heredoc delimiter";
        "[scanner-refusal:2af3620c848715087c00635d863574a5] innermost open construct";
        "[scanner-refusal:d7a6bb6f6a525cf9994430bc39b6954e] closes no subshell";
        "[scanner-refusal:4c983091dfcac75f21cacf41beaa556d] still open at";
        "[scanner-refusal:bf12194d221f2281dedd171e418892fa] heredoc fixture";
        "[scanner-refusal:f23aa9a38a34c91eadc946f44d66d28a] unsupported shell lexical";
        "[scanner-refusal:d06f79ba9c87a43873089eb160369d9c] unsupported shell execution";
        "[scanner-refusal:f472d334deac706e33952da7e799b22a] never turns errexit";
        "[scanner-refusal:1b6c2569972a33c6d8cdd0f5ea2c5752] a case inside";
        "[scanner-refusal:77db671c892d42f5738f054827480452] save-and-restore function is";
        "[scanner-refusal:b8ca55024e0a84b39a4011db683a7ba9] functions leave errexit";
        "[scanner-refusal:671f27a3cd7fa4d4c5dcb8af868a63fb] carries no whole-file";
        "[scanner-refusal:090dba89fb927b42ec1f1c29354ae999] the whole-file reading";
        "[scanner-refusal:88b12da9841dbf6131676bf4fb0f4686] statement-position ! command";
        "[scanner-refusal:003eba7381c87352fb02cd374fa62ea9] errexit-negation fixture";
        "[scanner-refusal:b67c631ab7ecd7df55fc002761cf99f6] reaches the absent()-style";
        "[scanner-refusal:b16ef3a2f724269c0b981a25871cbbb3] statement-position A &&";
        "[scanner-refusal:2d086d94d5a797eb9901008d71d5c241] errexit-and-list fixture";
        "[scanner-refusal:3f096017191c7a4d9f63d7a94dcfb6ff] reaches the one-predicate-per-statement";
        "[scanner-refusal:36dfd5919c0215c23122a0ec198d2739] errexit execution control";
        "[scanner-refusal:c4194f68155e0665e0176c65824cb4fa] trap-order fixture";
        "[scanner-refusal:624b5e03b7974c6da8261c5b00b3f94f] reaches the trap-order";
        "[scanner-refusal:7dddd02c97cea5d19abe99b74fa55dee] reaches the unread-signal";
        "[scanner-refusal:bd11a3abd96cb5a9802b6d3d4246e404] reaches the unread-structure";
        "[scanner-refusal:d6f4a25fb971a1f7f1c33b8fa3a42a54] script whose structure";
        "[scanner-refusal:4b1c2d6d0d37de04d679a9d0323a5467] trap-order execution control";
        "[scanner-refusal:730f080d2e13414d67084e34416888e9] takes its lifecycle";
        "[scanner-refusal:711cc6b5aa70f8e1ec411351c89c6ccc] lifecycle or redefines";
        "[scanner-refusal:348d853f9ff7699418e79b2c7469d193] the command running";
        "[scanner-refusal:beeb940e51aa3dff752296c0bd236146] lifecycle are accepted";
        "[scanner-refusal:0d15e8eb18e96635004da2561ab0917c] harnesses are discovered";
        "[scanner-refusal:2a0fdaf225cc691ffc63804ceb591d29] production test-run is";
        "[scanner-refusal:fd1b281d075e1e22e4fe44bc8642e4c7] duplicate harness";
        "[scanner-refusal:523d1f69c7bf5fef964dbf38dc8de10c] comment mentioning support";
        "[scanner-refusal:ac074b168e19f7f99423326fcdea5669] shebang";
        "[scanner-refusal:8743bc48ff2a3da0c15e40f654ad6ff7] shebang";
        "[scanner-refusal:3184b30e46c2d310e18bbae7c1c8ae9e] shared harness contract";
        "[scanner-refusal:15ee2c44c18fee32abb92315075e99ab] parses";
        "[scanner-refusal:f574a9b9751d21a02afcc893a83b2dbe] below the repository's";
        "[scanner-refusal:8d61869dcd0de680f0aead692a642ebe] the scan reached";
        "[scanner-refusal:8fe3f688c6c4ef4b004016098ba03f34] reached the session";
      ] );
    ( "test_config_consistency.ml",
      [
        "[scanner-refusal:40f6a4df02a115c327391e051bacd5c6] call-site keys missing";
        "[scanner-refusal:dfd9b6901bf188f6500bcc1284778dcd] known_config_keys";
        "[scanner-refusal:a22f67563f6e58a972758cfb6438b236] known_config_keys";
        "[scanner-refusal:f183d4fa8451402727499389f65d55a4] known_config_keys";
        "[scanner-refusal:9a0c38727d642eb678b4707766f25708] known_config_keys";
        "[scanner-refusal:1ce67444f994f72961fbc476c23dcd84] sets keys missing";
        "[scanner-refusal:2651df7744050e9d7f649f390b7c53b1] payload quoted in";
        "[scanner-refusal:3a6832937329cda23c716c3883e04cb2] has no '";
        "[scanner-refusal:8bb76ad7cf948a8181b88ee36b0148c6] scanned files share";
        "[scanner-refusal:11c8ffb35753ffb6f87da20d2ba6654f] Utils.settings";
        "[scanner-refusal:c56af001748a614d221709bf11098511] the empty string";
        "[scanner-refusal:e583c416783b3632963a66636f20f65e] string literal in";
        "[scanner-refusal:b4c59d5ad7a9c4e591998fccf8f039aa] exempted functions that";
        "[scanner-refusal:4e4d383baae3810309379fe0b2c3e9ef] registered default goes";
        "[scanner-refusal:18e6c65c27f35f658858a12b7c8dfd72] literal default is";
        "[scanner-refusal:e72f69460ef84de5e85f6c2bbe7bef5b] docs/agent-notes/build-and-test.md";
        "[scanner-refusal:55743888dd2a4a2234ee1d3f031ea47e] meets its source-count";
        "[scanner-refusal:2314f0c5d2762ecb0069dfb64d5c8a54] known_config_keys";
      ] );
    ( "verdict_ratchet.ml",
      [
        "[scanner-refusal:977473ba169d481e2b57377845e9a8b8] exempted quantified bindings";
        "[scanner-refusal:52b0c212a9f1ba97a1400cd478d7cfda] shadowing definitions separate";
        "[scanner-refusal:28926aa707bb0e8c7e5a087d40f0bf13] exempted claim-shaped literal";
        "[scanner-refusal:b0f5d68f72c0eeaca59c45fd980450cd] reports the claim";
        "[scanner-refusal:138941bb9d8e8476a0e47662d3b84393] reports the claim";
        "[scanner-refusal:b58d64bebca3825149316fa29084d773] which reads";
        "[scanner-refusal:06f7a9a77b49e173f538bb4f1b062a58] gate reader raised";
        "[scanner-refusal:0de32014cb53ae9628faaa035dbb0ac7] starting with";
        "[scanner-refusal:6e412a3ea539daeb755ee0d61f3d40f7] exempt_skip_lists";
        "[scanner-refusal:fc367fa3dd3580b30c61063bebf7bd25] applies the leg";
        "[scanner-refusal:3b5509d5f78cc3d75c4b9816b7f0021a] exempt_uncompared_legs";
        "[scanner-refusal:cccb6b32fe71c4e65b2b51a39ea8c519] skip-list reader raised";
        "[scanner-refusal:f3e306858423827e6575d13a17d57af6] check cannot vouch";
        "[scanner-refusal:4b8222a751df16e2ca351f7bacca5583] claim readers raised";
        "[scanner-refusal:6fa8a8891ce1cc5457a70a7dc0955684] let`-bound partial application";
        "[scanner-refusal:f94dbc9f97cc1e9b0662107356001f6b] applies the leg";
        "[scanner-refusal:1decd3d7c917569b94d9f613373fc1c2] population or gate";
        "[scanner-refusal:41f6b5db55cef6f0486ec091a293a838] Verdict_scan.leg_gate_allowed_paths";
        "[scanner-refusal:c921c1a193838de76e92d91d550cd7b1] executes";
        "[scanner-refusal:df4e110ae1379f998d366f0875cbade7] exempt_leg_gate_calls";
        "[scanner-refusal:deafc7763427d518f93ceafc49426fa1] leg-gate independence reader";
        "[scanner-refusal:3bd156674aa90688bc0e75de9fb6c146] stdout probe token";
        "[scanner-refusal:c2cbc4e8a5802e3ad11a4c22d434e95b] run-constant gate into";
        "[scanner-refusal:bff7d0c903bdd22694f18ebeaf0d916f] folded-gate reader raised";
        "[scanner-refusal:add5883320d9cdf491a628c74ff3aa98] folds the run-time";
        "[scanner-refusal:0599e4e661e8f3457de44f5cb1b23a1a] compares float readbacks";
        "[scanner-refusal:e10fe49d52c241506e850ae9e6766dc9] tolerance reader raised";
        "[scanner-refusal:1045125e8f0a2a1ed0a5933a3c7ab89e] hand-rolled success trailer";
        "[scanner-refusal:085cfc1230ac1435b3930c2de950d243] passes `~min";
        "[scanner-refusal:fccffb542fab0ec734024940e42605cc] exempt_floors";
        "[scanner-refusal:4dde956f3421b7ae6e9cff871b40d8b4] floor reader raised";
        "[scanner-refusal:0f307832357ee7ca58b9798a86530dac] arguments the rule's";
        "[scanner-refusal:53b71ea33a69d24c941985f4eb35406b] prints the claim";
        "[scanner-refusal:7d1962ff34bbe1ed7b89a535dc0cfd7e] exempted literals that";
        "[scanner-refusal:ad023891a3bd7827b3905292c487055b] claim-shaped literal any";
        "[scanner-refusal:e1354d390a356f0c79420c25232b0b42] planted canaries the";
        "[scanner-refusal:b3bbed986e758647a2db67f9428972c6] claims through Verdict";
        "[scanner-refusal:63d9475f3df44af6baac8bdc03a3310b] every quantified binding";
        "[scanner-refusal:d4d40c92b012f9347593f527cb7ad30d] pass_fail";
        "[scanner-refusal:5378ba5390debc4a8425ccdeb107f2d1] pass_fail";
        "[scanner-refusal:df1860b788314bd677ee963aff3f8186] label literal carries";
        "[scanner-refusal:7005fe842a74ccccd962cdda19ae98be] line-break reader read";
        "[scanner-refusal:b56027d13651cbbf21ae177b43426575] parity claim hand-rolls";
        "[scanner-refusal:d36bcdb919d2b79b23dc1e15969fde60] tolerance reader read";
        "[scanner-refusal:084b5e19d97705d191e28ba889a52530] test source hand-rolls";
        "[scanner-refusal:fd2fc8de1719682e3cecae854e708253] trailer reader read";
        "[scanner-refusal:2d867210446f1449a6beebc663d207c6] Verdict.Claims";
        "[scanner-refusal:b03ecbfa5a4a006730722bb2e47ba011] Verdict.Claims";
        "[scanner-refusal:20d88fc03901bcbdfbc113ed14029a7a] Verdict.Claims";
        "[scanner-refusal:43f45dea7e13ae71ff6e05054baafcca] quantified claim passes";
        "[scanner-refusal:e7253c1a7a914e2fb31ae0576e3b56eb] the literal floors";
        "[scanner-refusal:72bcef9d9ddc2338fe8af8ef65990041] Verdict.Claims";
        "[scanner-refusal:b42517e6f02cc3aa2ca5691ebdd70c62] population it quantifies";
        "[scanner-refusal:ccb1c4138577d668c45762104c7e40a4] Verdict.Claims";
        "[scanner-refusal:a51e429cf1070db9d057a59d7b50c801] reader completed gated";
        "[scanner-refusal:03571cf329d0733bb946c2eca2946008] Verdict.Claims";
        "[scanner-refusal:f675a65703076dfc53ea44fb59f9fd09] claim-name resolver resolves";
        "[scanner-refusal:664fc647996773a4adb1a331fc223f1f] gates and populations";
        "[scanner-refusal:578e6b63483d4d08d00475e2d75ee60b] exempt_uncompared_legs";
        "[scanner-refusal:e06e04779aed13eb61abbdbd6392f9c4] exempt_leg_gate_calls";
        "[scanner-refusal:b7208bf2fcc5db24ceb15a5611291be8] in_leg";
        "[scanner-refusal:74964ef6b574e06b674507a8e449d7e2] independence reader inspected";
        "[scanner-refusal:92b2164afd41d9fd93850d02494032d7] run-constant gate into";
        "[scanner-refusal:e1194c69948ad1ab8eca8dfe687e1962] run-time hatch into";
        "[scanner-refusal:90b4edf499fe49feb30dde99ff5e87b3] no source writes";
        "[scanner-refusal:83059def842c32ad5e0ce705c42e5d69] skip-list reader read";
        "[scanner-refusal:e9935bd293c26e123652183c000cb7e8] Verdict.Claims";
        "[scanner-refusal:07f11cb8fe0952beedc14df2223a8f0f] values of Verdict";
        "[scanner-refusal:be24b406399ca10a3226fec87a6d2b7f] every literal planted";
        "[scanner-refusal:7b69f1ae79d020bf8ce24352f17dec6d] every exemption on";
        "[scanner-refusal:bd1c6476261fa5d1b680ca8c0959f3f9] exempted claim-shaped literal";
        "[scanner-refusal:e29d881451dd7fd465b12ea6911fc165] exempted quantified binding";
        "[scanner-refusal:0c4e9524d3ca5b65bcbff10057237e01] read string literals";
        "[scanner-refusal:52611c0a0d560ef9a05c306005ef3e2a] them as arguments";
        "[scanner-refusal:d50daada9fe28bdf45d8d12250595a40] one test directory";
      ] );
    ( "workflow_job_timeouts.ml",
      [
        "[scanner-refusal:ad8d2fa1c69a86cbbb71c86de6fa1311] checked-in GitHub workflow";
        "[scanner-refusal:5c0ba3bcf03da17c20f587f737d8937b] job-level runtime ceiling";
        "[scanner-refusal:60c7ecc729f54888b5505b2a62d74bc5] workflow job states";
        "[scanner-refusal:a1cbed1bccff6a63840ed1fcbe8731d7] docs/agent-notes/build-and-test.md";
        "[scanner-refusal:0615ba230c2da562231c908bacb9d100] expected_sha-dispatched";
        "[scanner-refusal:1d23389c61196c5055abc91e2b7abe75] bisection-floor record lists";
        "[scanner-refusal:406e6c8f86b40c87a97b7caa4d7a79cf] script runs without";
        "[scanner-refusal:e2fcdedf2103db72a9b3c2b933471215] reusable-workflow call passes";
        "[scanner-refusal:d19a5eafc27ec268076c5d7ca7dbe979] GitHub's 6-hour default";
        "[scanner-refusal:5df256dd2c78f0fb5ea7e3a3c757c976] ceiling declared after";
        "[scanner-refusal:ac061fc23855d36465a9bd2ac3df8a14] ceiling that exists";
        "[scanner-refusal:42f8936ed74b6847e68661e48376b771] per-step ceiling is";
        "[scanner-refusal:77aa0c14bff72954feb09cf4c386a264] reusable-workflow call that";
        "[scanner-refusal:9f23eda281ea0c55a77c9e409e9471f6] timeout-minutes key with";
        "[scanner-refusal:fbfd4638215ae05bcb97101b459be8e4] timeout-minutes key whose";
        "[scanner-refusal:186c72d23e1ef379b2b9d5bd606fc1aa] real ceiling wearing";
        "[scanner-refusal:7d12059cdf4ae071e9bf2d7e0576b7ae] reusable-workflow call carrying";
        "[scanner-refusal:9a63876ae18ab76d104686433857254d] workflow is discovered";
        "[scanner-refusal:c8c51a8f093266f7fc1d691b313d9558] steps sequence starts";
        "[scanner-refusal:9c0d4e8cb577c3910eed5b4e88a437de] reader cannot decide";
        "[scanner-refusal:a4809b85dfbba2c0a2c42ed01d417e17] refused rather than";
        "[scanner-refusal:90eaa52d68d4ad193b704ab3d1946e09] keys being unreadable";
        "[scanner-refusal:78e566c471ad4160d6823ebd5e6bf4d0] refused rather than";
        "[scanner-refusal:06c1a18e5c944d59c7a4558c689db819] unreadable line inside";
        "[scanner-refusal:781d64c7cd92fd55d5eb911b61f21dbe] key spelled plainly";
        "[scanner-refusal:028700f84c8213b498760b2e7f50530c] bounded job passes";
        "[scanner-refusal:aa6e1fd6dbb8b02a479bd6e8ca1f9466] quoting hides nothing";
        "[scanner-refusal:62c5c78cf77044d66c80af05a68c6cea] sequence entry where";
        "[scanner-refusal:7607f0476a95ee77b5b691412ab32180] body staying content";
        "[scanner-refusal:2fff1ce44fe968a0f0b0a8600c33dc29] indented root mapping";
        "[scanner-refusal:5175d9278abfe05422915dbfd01d2789] multiline quoted scalar";
        "[scanner-refusal:78a687bff8c4929259602012af00edb7] same field written";
        "[scanner-refusal:410e5f2257be17b30c49cbabd3debfb3] an apostrophe inside";
        "[scanner-refusal:b2e9a89191c1483a6cbafac957165abc] before it notwithstanding";
        "[scanner-refusal:0dcbc70ff6b0ed1e7ccb8334ea44f89f] scalar indicates nothing";
        "[scanner-refusal:833ecfa7cd822e9a1a7f177bbf9dc3fe] carrying YAML's separation";
        "[scanner-refusal:b1d9df243c7ef140743dbb4b09a54c86] leading byte-order mark";
        "[scanner-refusal:4ed00663149579fcce944aba3b286b51] document-end marker leaves";
        "[scanner-refusal:d41b6151de5f78d82eecb470d7ce97d4] its workflow passes";
        "[scanner-refusal:f1d6bc11067a6db9ea16f25c8e526f98] carrying YAML's indentation";
        "[scanner-refusal:63a7ea671879e6eb8a95504d21ae4856] document indents uniformly";
        "[scanner-refusal:501e8a68f00d9f02dc169f32d4980a9f] reader measuring indentation";
        "[scanner-refusal:de56c44843177e1b1a86d6d0970e8ec9] refused rather than";
        "[scanner-refusal:ea2e7502608bb8e1195b7ec15868624a] nested under github/workflows";
        "[scanner-refusal:a557763ee03b78a87affc426156201bd] bisection-floor record does";
        "[scanner-refusal:c673d72de1101d58abd6fa2f4919dca8] action without moving";
        "[scanner-refusal:011823d18c83a1683a6ff1b57e7825a6] moving the floor";
        "[scanner-refusal:b8e01bb6fdcc84fd7c2e1ff184f818cc] entry wrapped across";
        "[scanner-refusal:b3b483b08fd6c863771924a83e02fc81] to it notwithstanding";
        "[scanner-refusal:57be0a3b386aed33500f9de691473802] and passes unlisted";
        "[scanner-refusal:8ff369030da0c9f0cc29b387b587a4af] job-level reusable-workflow call";
        "[scanner-refusal:d06f17413610f7c10892e31c144519ff] expected_sha";
        "[scanner-refusal:4dbaace6e8cda6a093513e7813af97d8] dispatched workflow's job";
        "[scanner-refusal:e45cc713244741cef9282e6770731b91] action refused unlisted";
        "[scanner-refusal:877cf35a722e5c81029167e73f54926f] flow-mapping step spelling";
        "[scanner-refusal:18bcee09d444d55d3cc60dbc9a241313] action no dispatched";
        "[scanner-refusal:4306dbe1b75e8925da9b4f3d5bf5b785] with their repetitions";
        "[scanner-refusal:0ad72760b2d1df6b8cf53b7485391ccf] refused never read";
        "[scanner-refusal:8763942273d7a3ad21572f947831fe2c] thirteenth month is";
        "[scanner-refusal:a81382ae52d8d269d1f22355d0bf21d1] entry in another";
        "[scanner-refusal:712f345271549ed4685da1a3cd914be3] entries are refused";
        "[scanner-refusal:97d4a9653aa56f3e6bf1d8cf7d273969] entry-shaped bullets unread";
        "[scanner-refusal:6fa59f2fcd334d1052d38d22c80f0a56] refused rather than";
        "[scanner-refusal:65f565042e69ddfbe03cc3c16b4ce893] index refreshes bounded";
        "[scanner-refusal:5425097dabc07fb6e6b617116a50408e] apt-get update with";
        "[scanner-refusal:778d0df15ba4633969ee5e75dfca77fd] bound the refresh";
        "[scanner-refusal:1d8676bb4dc33dec91e877e77b2bb3e4] duration which disables";
        "[scanner-refusal:6c6f804218c9e80bc1011ef8b235342a] separator before the";
        "[scanner-refusal:25a0771237987c37fd4f7b8481290619] apt's options between";
        "[scanner-refusal:fc33c8d3ecfe78254790472f5fed3b0a] front end's refresh";
        "[scanner-refusal:b2d928c764e587d21c62d0d1bdfaaa22] unbounded refresh quoted";
        "[scanner-refusal:fe8df2a7976b0441fb78a41a8ee07af0] command substitution inside";
        "[scanner-refusal:18034cc42c5fded559e7a29c1c57a615] script's unbounded refresh";
        "[scanner-refusal:6b6eff6eb802894930d7281d3548beb2] composite action's unbounded";
      ] );
  ]

let entries =
  List.map raw_entries ~f:(fun (source, markers) -> ("test/operations/" ^ source, markers))

(** The markers registered for [source], none for a source without a row: the catalogue names a
    source outside [sources] itself (gh-ocannl-1088) and still reports its row difference. *)
let markers source = Option.value (List.Assoc.find entries source ~equal:String.equal) ~default:[]

let sources = List.map entries ~f:fst

let raw_direct_evidence =
  [
    ( "env_var_deps.ml:57263e03b8b032365fbd586bbd2705a3",
      "a captured Verdict run is refused when only a sibling run clears the variable: true" );
    ( "codegen_text_inventory.ml:123bedafe19167d3536ddb2ca1115d21",
      "a scan handed a source that does not parse refuses it, exiting 1: true" );
    ( "codegen_text_inventory.ml:532d3ed986dd6559fec87d780769ee1f",
      "a scan handed a wrapper interface without its members' interfaces refuses it, exiting 1: \
       true" );
    ( "env_var_deps.ml:05b0b536b88908d4ef277447d7e82ae3",
      "an exemption whose read disappears is refused: true" );
    ( "env_var_deps.ml:95919586e2a2b25eaaf13ca8093ab584",
      "a repository-wide scan rule running nothing this check can name is refused: true" );
    ( "env_var_deps.ml:be41f253ce4b808b6b4ec587c304ec08",
      "a repository-wide scan rule running a script rather than an executable is refused: true" );
    ( "env_var_deps.ml:b040632469771b25abb3469878413be3",
      "a repository-wide scan rule running an executable whose source the check was not handed is \
       refused: true" );
    ( "config_dep_completeness.ml:2ade05f14bd972f85a1792722f8a0ab6",
      "config_dep_completeness refuses a run handed no dune file, exiting 1: true" );
    ( "config_dep_completeness.ml:6c29eb9b1db92bba229707a7cd8e1b5b",
      "config_dep_completeness refuses a tree built to trip its refusals, exiting 1: true" );
    ( "config_dep_completeness.ml:cebf445823df462f70b459a86b20e3d3",
      "config_dep_completeness refuses a tree built to trip its refusals, exiting 1: true" );
    ( "config_dep_completeness.ml:5e382153a89be91293964b5417659f1a",
      "config_dep_completeness refuses a tree built to trip its refusals, exiting 1: true" );
    ( "config_dep_completeness.ml:6d151bed80a45201e61d096c2c38d210",
      "config_dep_completeness refuses a tree built to trip its refusals, exiting 1: true" );
    ( "config_dep_completeness.ml:5509acc7fac8ff0412e3631b49c06dc7",
      "config_dep_completeness refuses a tree built to trip its refusals, exiting 1: true" );
    ( "config_dep_completeness.ml:e11f1dd0ebe8035f6b6bb919c904e611",
      "config_dep_completeness refuses a tree built to trip its refusals, exiting 1: true" );
    ( "config_dep_completeness.ml:e20c1b4ad3881b8147939aef02357ab2",
      "config_dep_completeness refuses a tree built to trip its refusals, exiting 1: true" );
    ( "config_dep_completeness.ml:332abda7feebdaa6c1c10670865ea5c6",
      "config_dep_completeness refuses a tree built to trip its refusals, exiting 1: true" );
    ( "config_dep_completeness.ml:328553bf93b184a97ab7d44d97c369a4",
      "config_dep_completeness refuses a tree built to trip its refusals, exiting 1: true" );
    ( "config_dep_completeness.ml:3e9cd231a7db9663f3ceba5ab3891105",
      "config_dep_completeness refuses a tree built to trip its refusals, exiting 1: true" );
    ( "config_dep_completeness.ml:cb948484fece727200c2dc74529ab671",
      "config_dep_completeness refuses a tree built to trip its refusals, exiting 1: true" );
    ( "config_dep_completeness.ml:86529712a3b8e4b7a66a3adcb8134331",
      "config_dep_completeness refuses an ambient record that leaves out the host, exiting 1: true"
    );
    ( "config_dep_completeness.ml:86529712a3b8e4b7a66a3adcb8134331",
      "config_dep_completeness refuses an ambient record whose producer depends on the host record \
       and never reads it, exiting 1: true" );
    ( "digest_completeness.ml:7cab0a5763f4811f331a4aab327c6784",
      "digest_completeness refuses the malformed tree, exiting 1: true" );
    ( "digest_completeness.ml:1825a277ba0eda3313fbe079e465c15d",
      "digest_completeness refuses the malformed tree, exiting 1: true" );
    ( "env_var_deps.ml:b0ff4d4e5709299b84685f0fc189cc63",
      "a rule whose alias names one module and whose run tests another is reported: true" );
    ( "env_var_deps.ml:5893c6332b1f903470ec51cad4f20e8f",
      "a tested module with no per-module rule is reported, and the checker exits 1: true" );
    ( "env_var_deps.ml:5f2b09efad6e397fb5bd77b84255d0bf",
      "a per-module alias listed in the `runtest` aggregate is reported as a double run: true" );
    ( "env_var_deps.ml:490ac5e3405d328c2953c38df4f0974b",
      "a run handed no dune file and no source is refused: true" );
    ( "env_var_deps.ml:8dbf084aae3ed7d3e4ce33fb0d06e710",
      "a run handed no deny-list is refused rather than passing every declaration: true" );
    ( "env_var_deps.ml:7aeca22f89e6555c3b97967f8bbdaba4",
      "a stanza declaring `(env_var GH_TOKEN)`, a name on the deny-list, is refused by name: true"
    );
    ( "env_var_deps.ml:a53c6b4b54cc85d2f251e80dd90c855c",
      "a module reading `Sys.getenv_opt \"GH_TOKEN\"`, under the passing tree, is refused by name: \
       true" );
    ( "env_var_deps.ml:034280d144ee4338406ff5f6a1cf287d",
      "a dune file built to trip the marker grammar and the declaration checks is refused: true" );
    ( "env_var_deps.ml:50ad2bbac7aab894b82d2525afd2aaf8",
      "a dune file built to trip the marker grammar and the declaration checks is refused: true" );
    ( "env_var_deps.ml:60a17b773827a71dd5c724af51013878",
      "a dune file built to trip the marker grammar and the declaration checks is refused: true" );
    ( "env_var_deps.ml:bd8470207f6c45ce4020275b78402293",
      "a dune file built to trip the marker grammar and the declaration checks is refused: true" );
    ( "env_var_deps.ml:78e6a1f5c4350e90995d5b6e874d7e3b",
      "a dune file built to trip the marker grammar and the declaration checks is refused: true" );
    ( "env_var_deps.ml:30c88c1631600d7969a9aefd4de74162",
      "a dune file built to trip the marker grammar and the declaration checks is refused: true" );
    ( "env_var_deps.ml:f2633aa4ebd6f2c33fd21e4d69cc9786",
      "a dune file built to trip the marker grammar and the declaration checks is refused: true" );
    ( "env_var_deps.ml:2803edcc7de84ea6d928343b9d8c8ad2",
      "a dune file built to trip the marker grammar and the declaration checks is refused: true" );
    ( "env_var_deps.ml:533fc9999ad0841992e0c1734d9c1f5d",
      "an external command handed a file this workspace builds is a stanza the rule reaches, \
       reported by name when it declares neither: true" );
    ( "env_var_deps.ml:7336dd5a712b0c3a578b64d7defb1654",
      "a dune file built to trip the marker grammar and the declaration checks is refused: true" );
    ( "env_var_deps.ml:1aaab6e3cd8f4615e1cd1d0321e309a1",
      "a dune file built to trip the marker grammar and the declaration checks is refused: true" );
    ( "env_var_deps.ml:bbadfd8513559355fedbc54dc3dc678a",
      "a `(subdir …)` group whose actions take the training lock and whose gate does not is \
       reported there too: true" );
    ( "env_var_deps.ml:b7991c01507b15cea041c6066f07038b",
      "the training lock spelled without its leading `/` is a per-directory lock, and is refused: \
       true" );
    ( "env_var_deps.ml:2a4ae5808191d606dc5b7b74dd4ef93e",
      "a lock name built from a pform cannot be compared across directories, and is refused: true"
    );
    ( "env_var_deps.ml:77d0be0fe2fca9a97da8be7f833d9c6e",
      "one relative lock name written in two directories is two locks, and is refused naming both: \
       true" );
    ( "env_var_deps.ml:eb5d5419b3bd18240469743cbabc58f2",
      "a rule on a `(test)`'s generated alias that runs ANOTHER directory's binary is not the \
       deliberate gate collision, and is reported: true" );
    ( "env_var_deps.ml:a06dccc27afe2153a7700609691d3c8d",
      "a dune file built to trip the marker grammar and the declaration checks is refused: true" );
    ( "env_var_deps.ml:7f157a5478f43aa3d2da34c9975976a9",
      "an unaggregated targetless source_tree scan reports that it inventories the repository: true"
    );
    ( "env_var_deps.ml:91751d43d8939e1da724e6f0bcbe8f09",
      "an undiffed source_tree output reports that it inventories the repository: true" );
    ( "env_var_deps.ml:f16784d0b1f52de5e30378cb771ada5c",
      "a runner that runs another directory's executable of the same name does not aggregate this \
       directory's member: true" );
    ( "env_var_deps.ml:4012b24c7623acd977522e8f2eacf2bb",
      "a rule reusing the alias dune generates for a `(test)` in the same `(subdir …)` group is \
       reported there too: true" );
    ( "env_var_deps.ml:ba5d90fc4bcec944ebd169329c706968",
      "a dune file built to trip the marker grammar and the declaration checks is refused: true" );
    ( "env_var_deps.ml:c1a398b587b5de3d7d02145dc45274a6",
      "a dune file built to trip the marker grammar and the declaration checks is refused: true" );
    ( "env_var_deps.ml:2df546d04b63387a24409b3c4c6a9611",
      "a tracing dependency without a module reading it is refused: true" );
    ( "env_var_deps.ml:638620a0745d0a136810c5b0256e0ec0",
      "each missing scoped exemption is stale even while its sibling alias remains: true" );
    ( "env_var_deps.ml:a81eacf691e2faeada3ee2b79d5fd7a6",
      "a blessed helper that stops stating a key, and a support value stating every key unblessed, \
       are both refused: true" );
    ( "env_var_deps.ml:0a10ab654cc394b5f797d88ea889d086",
      "a marker between stanzas is refused rather than attributed: true" );
    ( "env_var_deps.ml:2dc2479687a000324b3f828912a7f4a8",
      "an entry point renamed away is refused as a lost anchor, not read as no search: true" );
    ( "env_var_deps.ml:677d819fac904c64b7025b4f17c07b5a",
      "a marker on a program that starts no search is refused as stale: true" );
    ( "env_var_deps.ml:22289447396ba1e1150aeaeb5faefadc",
      "a program starting a search with a run that can select cc is reported when it neither pins \
       the width nor carries the marker: true" );
    ( "env_var_deps.ml:fa823e2140df2fbae2d15be1794a146b",
      "a stanza that can select cc and runs a command naming no program in command position -- a \
       shell line, a PATH tool handed an executable of ours -- is refused, not read as running \
       none: true" );
    ( "env_var_deps.ml:9de3b3fd1cd96a8d87e76bfa31534244",
      "a marker declaring anything but `independent` is refused: true" );
    ( "env_var_deps.ml:a45f7f9b519fd1f5ef3516593179eef6",
      "a marker on a program that pins the width is refused as contradictory: true" );
    ( "env_var_deps.ml:be4bb5e070f7c1653c2f7236cb2e6a18",
      "the marker's sentinel outside a comment is refused: true" );
    ( "env_var_deps.ml:e47faeb8aa90e38e3dd7b698f6ac8998",
      "two markers on one stanza are refused: true" );
    ( "env_var_deps.ml:e96ac262e859176f6f0ec05d6f2f30cb",
      "a marker on a rule, which names no program's modules, is refused: true" );
    ( "test_config_consistency.ml:40f6a4df02a115c327391e051bacd5c6",
      "test_config_consistency refuses the malformed tree, exiting 1: true" );
    ( "test_config_consistency.ml:dfd9b6901bf188f6500bcc1284778dcd",
      "test_config_consistency refuses the malformed tree, exiting 1: true" );
    ( "test_config_consistency.ml:a22f67563f6e58a972758cfb6438b236",
      "test_config_consistency refuses the malformed tree, exiting 1: true" );
    ( "test_config_consistency.ml:f183d4fa8451402727499389f65d55a4",
      "test_config_consistency refuses the malformed tree, exiting 1: true" );
    ( "test_config_consistency.ml:1ce67444f994f72961fbc476c23dcd84",
      "test_config_consistency refuses the malformed tree, exiting 1: true" );
    ( "test_config_consistency.ml:2651df7744050e9d7f649f390b7c53b1",
      "test_config_consistency refuses the malformed tree, exiting 1: true" );
    ( "test_config_consistency.ml:3a6832937329cda23c716c3883e04cb2",
      "test_config_consistency refuses the malformed tree, exiting 1: true" );
    ( "test_config_consistency.ml:8bb76ad7cf948a8181b88ee36b0148c6",
      "test_config_consistency refuses the malformed tree, exiting 1: true" );
    ( "test_config_consistency.ml:11c8ffb35753ffb6f87da20d2ba6654f",
      "test_config_consistency refuses the malformed tree, exiting 1: true" );
    ( "test_config_consistency.ml:c56af001748a614d221709bf11098511",
      "test_config_consistency refuses the malformed tree, exiting 1: true" );
    ( "test_config_consistency.ml:e583c416783b3632963a66636f20f65e",
      "test_config_consistency refuses the malformed tree, exiting 1: true" );
    ( "test_config_consistency.ml:b4c59d5ad7a9c4e591998fccf8f039aa",
      "test_config_consistency refuses the malformed tree, exiting 1: true" );
    ( "verdict_ratchet.ml:0f307832357ee7ca58b9798a86530dac",
      "verdict_ratchet refuses a run handed no source, exiting 1: true" );
    ( "verdict_ratchet.ml:f3e306858423827e6575d13a17d57af6",
      "verdict_ratchet refuses a corpus built to trip its refusals, exiting 1: true" );
    ( "verdict_ratchet.ml:53b71ea33a69d24c941985f4eb35406b",
      "verdict_ratchet refuses a corpus built to trip its refusals, exiting 1: true" );
    ( "verdict_ratchet.ml:7d1962ff34bbe1ed7b89a535dc0cfd7e",
      "verdict_ratchet refuses a corpus built to trip its refusals, exiting 1: true" );
    ( "verdict_ratchet.ml:ad023891a3bd7827b3905292c487055b",
      "verdict_ratchet refuses a corpus built to trip its refusals, exiting 1: true" );
    ( "verdict_ratchet.ml:e1354d390a356f0c79420c25232b0b42",
      "verdict_ratchet refuses a corpus built to trip its refusals, exiting 1: true" );
    ( "verdict_ratchet.ml:6e412a3ea539daeb755ee0d61f3d40f7",
      "refuses a leg claim it cannot complete, a planted hand-written skip list, a skip-list \
       reader that raised, and a stale exemption: true" );
    ( "env_var_deps.ml:02e183a4105132ec092206c1d35d6247",
      "a dune file built to trip the marker grammar and the declaration checks is refused: true" );
    ( "env_var_deps.ml:0deef0cb94c45615f32ee84ac2e76f8e",
      "a dune file built to trip the marker grammar and the declaration checks is refused: true" );
    ( "env_var_deps.ml:e55f77a62871fb685d23199e95c8f173",
      "a dune file built to trip the marker grammar and the declaration checks is refused: true" );
    ( "env_var_deps.ml:92d872e8d8f1d6a3de97bf3499a7170d",
      "the checker exits 1 and names a key the ambient record carries, declared inside a `(subdir \
       …)` group: true" );
    ( "env_var_deps.ml:b9c87589aecbe6de4db58fc4671f7491",
      "a dune file built to trip the marker grammar and the declaration checks is refused: true" );
    ( "env_var_deps.ml:eac17cf643d980bb15c3113087c863c4",
      "a dune file built to trip the marker grammar and the declaration checks is refused: true" );
    ( "env_var_deps.ml:591d750b2330cdac197127abc39ce397",
      "a dune file built to trip the marker grammar and the declaration checks is refused: true" );
    ( "env_var_deps.ml:f30ace350fdf01fcc307ea64e391a7bf",
      "a dune file built to trip the marker grammar and the declaration checks is refused: true" );
    ( "verdict_ratchet.ml:1045125e8f0a2a1ed0a5933a3c7ab89e",
      "verdict_ratchet refuses a corpus built to trip its refusals, exiting 1: true" );
    ( "env_var_deps.ml:6d4652d347d64362ce0c9b3c1498abb9",
      "backend-default marker on wrong stanza is refused: true" );
    ( "env_var_deps.ml:eeabf0209c60b38c2cf243f1fcbaac4a",
      "backend-default marker outside any stanza is refused: true" );
    ( "env_var_deps.ml:f75580a3f9d63e77e8203a3df984bb9c",
      "backend-default sentinel outside a comment is refused: true" );
    ( "env_var_deps.ml:942031ff44cb018ae2d7eee70ed6327a",
      "duplicate backend-default reader on one stanza is refused: true" );
    ( "env_var_deps.ml:ffa1cdde8b0746fcdb7a1ac0b406107a",
      "duplicate backend-default definition is refused: true" );
    ( "env_var_deps.ml:2d95d96433c72c5c970b888087a59832",
      "backend-default definition with zero readers is refused: true" );
    ( "env_var_deps.ml:6c8e61d2480bca6f93d3fd99b945e5ad",
      "backend-default definition naming a missing module is refused: true" );
    ( "env_var_deps.ml:04f8a4dfd1058a7595117f80bc1fcb0c",
      "backend-default definition naming an unparseable module is refused: true" );
    ( "env_var_deps.ml:66742d7bb3766af11b75ab38bf798fdb",
      "backend-default definition naming a missing top-level definition is refused: true" );
    ( "env_var_deps.ml:b0d8f5f604f61eb8ecc4829d93f24a83",
      "backend-default reader naming an undefined default is refused: true" );
    ( "env_var_deps.ml:afd2f461af706b40bf4582f1b0a30b01",
      "backend-default reader whose backends do not overlap the default is refused: true" );
  ]

(** Direct failures no control executes, each with the reason none does: a refusal the scanner can
    reach only through a library constant, or only through a malformed repository no control builds.
    Such a refusal is still catalogued -- [print] lists it under its own heading, so the catalogue
    audit finds its marker -- but never as exercised: a positive fixture, a prose line or an
    unrelated claim is not evidence that a refusal path runs (gh-ocannl-1207). A key here and in
    [raw_direct_evidence] at once is refused by the catalogue audit, and one a control does execute
    is refused by [print]. *)
let raw_catalogue_only : (string * string) list =
  [
    ( "explicit_library_deps.ml:79b83d87bc35a637a45c0dd6dc7479cf",
      "decided by the argument vector its own dune rule writes" );
    ( "unused_compiler_deps.ml:1262ba0367f56ab63b0ef45185bb1257",
      "a defect in the check's own synthetic control text, not a malformed tree" );
    ( "unused_compiler_deps.ml:351ce2a461cb37393c61069ee2649d96",
      "decided by the argument vector its own dune rule writes" );
    ( "unused_compiler_deps.ml:36778054d93aa08eac389989017aed7a",
      "decided by dune, which sets DUNE_SOURCEROOT for every action; unset only outside the rule" );
    ( "digest_completeness.ml:fc3ef3788212f2d1f5a9d9a46be3ca38",
      "decided by the library constant `Utils.config_key_classification` alone" );
    ( "digest_completeness.ml:dc9995a5d5b80370dc90971d3fd019df",
      "decided by two library constants, `Utils.known_config_keys` against \
       `Utils.config_key_classification`" );
    ( "digest_completeness.ml:a385ae718c8af3ab39a45701b75fda4d",
      "decided by two library constants, `Utils.config_key_classification` against \
       `Utils.known_config_keys`" );
    ( "digest_completeness.ml:14b02d2180d3814d6746acd827aecbc3",
      "decided by two library constants, `Utils.config_key_classification` against \
       `Schedule_cache.key_components`" );
    ( "digest_completeness.ml:8109160f59717560ff5e6f7251a6a2b2",
      "decided by two library constants, `Schedule_cache.key_components` against \
       `Utils.config_key_classification`" );
    ( "test_config_consistency.ml:9a0c38727d642eb678b4707766f25708",
      "decided by two library constants, `Utils.profile_payloads` against `Utils.known_config_keys`"
    );
    ( "env_var_deps.ml:f11c09a8318b3a7863444256973f57d4",
      "a disagreement between this scan's two readers of one dune file: the raw-text floor exists \
       to catch the structural walk missing a stanza, and any input that trips it is a walk \
       defect, not a malformed tree" );
    ( "env_var_deps.ml:941a9fb77d867c7b1576a0f19ee5db3b",
      "a declaration outside every `deps` field; the scan's own reader of those fields is what \
       would have to miss it" );
    ( "env_var_deps.ml:25657f61eca857f6fd2f34de66e81eaf",
      "the live orphan refusal; `refusal_control` puts the relationship to a synthetic corpus in \
       process, past this branch" );
    ( "env_var_deps.ml:fe4c53f899eb08a9ae67e7a797a5a841",
      "stale entries of `refusal_exemptions`, which is empty" );
    ( "env_var_deps.ml:09ce44850cdf45c95e308d6c64aed813",
      "a floor under the repository's own census, which only a repository that lost its members \
       reaches" );
    ( "config_dep_completeness.ml:c35e7bf895b2bfc334caa198f2db4c83",
      "a disagreement between this scan's two readers of one dune file: the raw-text floor exists \
       to catch the structural walk missing a stanza, and any input that trips it is a walk \
       defect, not a malformed tree" );
    ( "config_dep_completeness.ml:a83ec7419ab530468bb3d4143c8c8576",
      "a disagreement between this scan's two readers of one dune file: the raw-text floor exists \
       to catch the structural walk missing a stanza, and any input that trips it is a walk \
       defect, not a malformed tree" );
    ( "config_dep_completeness.ml:28c35655d1568c3c11bdc1893d8847fa",
      "a disagreement between this scan's two readers of one dune file: the raw-text floor exists \
       to catch the structural walk missing a stanza, and any input that trips it is a walk \
       defect, not a malformed tree" );
    ( "config_dep_completeness.ml:929a18d24ebaad373970a360298e3b88",
      "a disagreement between this scan's two readers of one dune file: the raw-text floor exists \
       to catch the structural walk missing a stanza, and any input that trips it is a walk \
       defect, not a malformed tree" );
  ]

let qualify = List.map ~f:(fun (key, value) -> ("test/operations/" ^ key, value))
let direct_evidence = qualify raw_direct_evidence
let catalogue_only = qualify raw_catalogue_only
let observed_failures = Hash_set.create (module String)
let failure_key ~source ~identity = source ^ ":" ^ identity

let observe_failure ~source ~format =
  let identity =
    Stdlib.Digest.string (Refusal_control_scan.normalize format) |> Stdlib.Digest.to_hex
  in
  Hash_set.add observed_failures (failure_key ~source ~identity)

(* Keep the format before substitution: a fixture argument can echo any sibling diagnostic. *)
type refusal = { format : string; message : string }

let capture_refusal finish fmt =
  let format = Stdlib.string_of_format fmt in
  Printf.ksprintf (fun message -> finish { format; message }) fmt

let fail fmt = capture_refusal Fn.id fmt
let with_message refusal message = { refusal with message }

module type FAIL = sig
  val fail : ('a, unit, string, unit) Stdlib.format4 -> 'a
end

let collect_refusals f =
  let recorded = ref [] in
  let sink : (module FAIL) =
    (module struct
      let fail fmt = capture_refusal (fun refusal -> recorded := refusal :: !recorded) fmt
    end)
  in
  let result = f ~fail:sink in
  (result, List.rev !recorded)

let check_failures ?(quiet = false) ~source ~formats refusals =
  let actual = List.map refusals ~f:(fun refusal -> refusal.format) in
  let matched = List.equal String.equal formats actual in
  if matched then List.iter actual ~f:(fun format -> observe_failure ~source ~format)
  else if not quiet then (
    eprintf "control expected refusal formats [%s], got [%s]\n" (String.concat ~sep:"; " formats)
      (String.concat ~sep:"; " actual);
    List.iter refusals ~f:(fun refusal -> eprintf "  %s\n" refusal.message));
  matched

let refused_with ?quiet ~source ~format f =
  let (), refusals = collect_refusals f in
  check_failures ?quiet ~source ~formats:[ format ] refusals

let accepted f =
  let (), refusals = collect_refusals f in
  List.iter refusals ~f:(fun refusal ->
      eprintf "control expected no refusal, got: %s\n" refusal.message);
  List.is_empty refusals

(** The [FAIL: ] lines of a child run's captured [output]: what it refused. *)
let refusal_lines output =
  String.split_lines output
  |> List.filter_map ~f:(fun line -> String.chop_prefix (String.strip line) ~prefix:"FAIL: ")

(** The refused runs of a scanner child process, by the scanner's source: the captured output of
    each run a control handed malformed input and the child refused -- exited 1. A run an exception
    stopped (2), or one that passed, is no refusal, whatever it printed. Feed it refused runs only.
*)
let refused_runs = Hashtbl.create (module String)

let observe_refused_run ~source output = Hashtbl.add_multi refused_runs ~key:source ~data:output

(** The refused runs each control claim made, by [source] and claim label (gh-ocannl-1233): a
    [raw_direct_evidence] entry is checked against the runs of the claim it names, not against every
    run of the scanner, so a mapping to a claim whose own runs never print the refusal is no
    evidence, however many sibling runs do. *)
let claim_runs = Hashtbl.create (module String)

let claim_key ~source ~claim = source ^ "\n" ^ claim

(** [attribute ~source ~claim outputs]: of [outputs], the captured output of child runs exactly as
    {!observe_refused_run} was handed them, those of runs [source]'s scanner refused are [claim]'s
    -- the runs it asserts on. An output no refused run produced (a run that passed, or one an
    exception stopped) attributes nothing. Call it beside the [Verdict] claim it names, with the
    claim's own label. *)
let attribute ~source ~claim outputs =
  let refused = Hashtbl.find_multi refused_runs source in
  List.iter outputs ~f:(fun output ->
      if List.mem refused output ~equal:String.equal then
        Hashtbl.add_multi claim_runs ~key:(claim_key ~source ~claim) ~data:output)

(** The refusal lines of the runs attributed to [claim] of [source]. *)
let claim_lines ~source claim =
  Hashtbl.find_multi claim_runs (claim_key ~source ~claim) |> List.concat_map ~f:refusal_lines

(** [with_tree files f] writes [files], each a relative path and its contents, under a fresh
    temporary directory, applies [f] to that directory, and removes the tree again: the malformed
    input a scanner's refusal control hands it. *)
let with_tree files f =
  let root = Stdlib.Filename.temp_dir "refusal_control" "" in
  let rec directories path =
    let parent = Stdlib.Filename.dirname path in
    if String.equal parent Stdlib.Filename.current_dir_name then []
    else parent :: directories parent
  in
  let made =
    List.concat_map files ~f:(fun (path, _) ->
        if not (Stdlib.Filename.is_relative path) then
          invalid_arg ("Refusal_control_manifest.with_tree: not a relative path: " ^ path);
        directories path)
    |> List.dedup_and_sort ~compare:String.compare
    (* Parents before children: a parent is the shorter path. *)
    |> List.sort ~compare:(fun a b -> Int.compare (String.length a) (String.length b))
  in
  List.iter made ~f:(fun dir -> Unix.mkdir (Stdlib.Filename.concat root dir) 0o755);
  List.iter files ~f:(fun (path, data) ->
      Out_channel.write_all (Stdlib.Filename.concat root path) ~data);
  (* The whole tree, not only the files written: a child that leaves something behind must not turn
     the cleanup into the exception that masks its result. *)
  let rec remove path =
    match Unix.lstat path with
    | { Unix.st_kind = Unix.S_DIR; _ } ->
        Array.iter (Stdlib.Sys.readdir path) ~f:(fun entry ->
            remove (Stdlib.Filename.concat path entry));
        Unix.rmdir path
    | _ -> Unix.unlink path
    | exception Unix.Unix_error _ -> ()
  in
  Exn.protect ~f:(fun () -> f root) ~finally:(fun () -> remove root)

(** Runs the scanner whose source is [source] as a child process, [exe] over [args], records what it
    refused through [observe_refused_run] as a run of [claim] ({!attribute}), and says whether it
    refused: exited 1, as Verdict ends a run with a failed check -- not 2, which is a run an
    exception stopped early. [claim] is the label of the [Verdict] claim this answers, which is the
    call this one is the argument of. *)
let refuses ~source ~claim ~exe args =
  let ((status, _, _) as run) = Fresh_process.run ~exe args in
  let refused = Poly.equal status (Unix.WEXITED 1) in
  if refused then (
    let output = Fresh_process.output run in
    observe_refused_run ~source output;
    attribute ~source ~claim [ output ])
  else Fresh_process.report ~label:source run;
  refused

let claim_exercises passed_labels diagnostic =
  let rec take_match before = function
    | [] -> None
    | label :: after ->
        if Refusal_control_scan.format_matches ~format:diagnostic.Refusal_control_scan.format label
        then Some (List.rev_append before after)
        else take_match (label :: before) after
  in
  take_match [] passed_labels

(** What a run shows of a direct failure. *)
type standing =
  | Exercised  (** Its caught branch was observed, or its control passed and its refusal ran. *)
  | Catalogue_only  (** Listed in [raw_catalogue_only] and executed by nothing in this run. *)
  | Executed_yet_catalogue_only  (** Listed there, and executed all the same. *)
  | Unexercised

(** How much static text [format] fixes: the more, the more specific. *)
let specificity format =
  List.sum (module Int) (Refusal_control_scan.static_runs format) ~f:String.length

(** Whether a refused child-run [line] is [format]'s own: [format] matches it, and no format among
    [rivals] -- the other direct failures of the same scanner -- that fixes strictly more text
    matches it too. A general format ([call-site keys missing from %s: %s]) would otherwise count a
    specific sibling's line ([call-site keys missing from known_config_keys registry: %s]) as its
    own, and a mutant silencing the general refusal would survive on the sibling's output. *)
let owns_line ~rivals ~format line =
  Refusal_control_scan.format_matches ~format line
  && not
       (List.exists rivals ~f:(fun rival ->
            specificity rival > specificity format
            && Refusal_control_scan.format_matches ~format:rival line))

(** The claim label a [raw_direct_evidence] entry names, without its [": true"]. *)
let evidence_label claim = Option.value (String.chop_suffix claim ~suffix:": true") ~default:claim

(** The claim labels every [direct_evidence] entry under [key] names, in table order: a key may have
    several entries, each naming one claim that answers for the refusal, and each is consulted
    (gh-ocannl-1572). *)
let evidence_labels direct_evidence key =
  List.filter_map direct_evidence ~f:(fun (entry_key, claim) ->
      Option.some_if (String.equal entry_key key) (evidence_label claim))

(** Whether the control claim [label] answers for the direct failure whose [format] it is checked
    against: the claim passed, and one of the lines of the runs it made ([claim_lines]) is the
    refusal's own ({!owns_line} against [rivals]). The one test {!standing} applies to the claim an
    entry names and {!answering_claims} to every passed claim. *)
let claim_answers ~passed_labels ~claim_lines ~rivals ~format label =
  List.mem passed_labels label ~equal:String.equal
  && List.exists (claim_lines label) ~f:(owns_line ~rivals ~format)

(** The passed claims of this run that answer for [diagnostic] ({!claim_answers}), in the order they
    passed, each once: the candidates a new direct failure's [raw_direct_evidence] entry can name
    (gh-ocannl-1571). *)
let answering_claims ~passed_labels ~claim_lines ~rivals diagnostic =
  let format = diagnostic.Refusal_control_scan.format in
  List.fold passed_labels ~init:[] ~f:(fun seen label ->
      if List.mem seen label ~equal:String.equal then seen else label :: seen)
  |> List.rev
  |> List.filter ~f:(claim_answers ~passed_labels ~claim_lines ~rivals ~format)

(** The [standing] of a direct failure [diagnostic] under [key], from the tables and this run's
    observations, so that the rule can be pinned on tables the repository does not have to contain.
    The refusal RAN when its caught branch was [observed], or when one of the scanner's [refused]
    child-run lines is its own ([owns_line] against the scanner's other formats, [rivals]). A
    [direct_evidence] entry names the control claim that answers for the refusal, and it counts only
    when that claim passed AND one of the lines of the runs THAT CLAIM made ([claim_lines], by the
    claim's label; gh-ocannl-1233) is the refusal's own: the claim alone is an accepted fixture as
    far as this refusal knows, the line alone is a refusal nothing asserted on, and a line from a
    sibling claim's run is a refusal the mapped claim never asserted on; printed output is no input
    at all. Every entry under [key] is consulted ({!evidence_labels}), and any one that counts is
    enough. A catalogue-only refusal that ran, by either route and in any run, is reported as such.
*)
let standing ~direct_evidence ~catalogue_only ~observed ~refused ~claim_lines ~rivals ~passed_labels
    ~key diagnostic =
  let format = diagnostic.Refusal_control_scan.format in
  let owned = List.exists ~f:(owns_line ~rivals ~format) in
  let ran = observed || owned refused in
  let answered =
    List.exists
      (evidence_labels direct_evidence key)
      ~f:(claim_answers ~passed_labels ~claim_lines ~rivals ~format)
  in
  match (List.Assoc.mem catalogue_only key ~equal:String.equal, ran) with
  | true, true -> Executed_yet_catalogue_only
  | true, false -> Catalogue_only
  | false, _ when observed || answered -> Exercised
  | false, _ -> Unexercised

(** The failed claim behind the marker of [diagnostic], one its source's row lists, in a run whose
    checks that failed are [failed] ({!Verdict.failed_checks}), as the reason the marker did not
    print -- [None] when no failed check stands behind it, so that its claim or refusal did not run
    (gh-ocannl-1499). A claim's marker stands for the claim its format matches; a direct failure's
    for the [direct_evidence] claims answering for it under [key], every failed one named (one its
    caught branch records has no claim behind it). Only a failure is named: a run prints a subset of
    the row's markers, the rest exercised in another control golden, so "did not run" is every
    passing run's normal state and is left for the reader of a golden diff to infer. Taking the
    table as an argument, as {!standing} does, lets the rule be pinned on tables the repository does
    not have to hold.

    Answers the reason with the failed checks left for the markers after this one. A claim's marker
    consumes the failure it names, as {!claim_exercises} consumes a passing execution: two identical
    claim formats are two claims, and one failure stands behind one of them, the other's claim not
    having run. A direct failure's marker consumes nothing: one control claim answers for several
    refusals, and its failure stands behind each. *)
let why_not_printed ~direct_evidence ~failed ~key diagnostic =
  let format = diagnostic.Refusal_control_scan.format in
  match diagnostic.Refusal_control_scan.kind with
  | Refusal_control_scan.Claim ->
      let rec take before = function
        | [] -> None
        | name :: after when Refusal_control_scan.format_matches ~format name ->
            Some (Printf.sprintf "its claim failed: \"%s\"" name, List.rev_append before after)
        | name :: after -> take (name :: before) after
      in
      take [] failed
  | Refusal_control_scan.Fail ->
      (match
         List.filter (evidence_labels direct_evidence key) ~f:(List.mem failed ~equal:String.equal)
       with
        | [] -> None
        | [ label ] -> Some (Printf.sprintf "the claim answering for it failed: \"%s\"" label)
        | labels ->
            Some
              (Printf.sprintf "the claims answering for it failed: %s"
                 (String.concat ~sep:", " (List.map labels ~f:(Printf.sprintf "\"%s\"")))))
      |> Option.map ~f:(fun reason -> (reason, failed))

(** The [raw_entries] row for [source] (its key without the [test/operations/] prefix) listing
    [diagnostics]' markers in extraction order, ready to paste. *)
let row ~source diagnostics =
  match diagnostics with
  | [] -> Printf.sprintf "    (%S, []);\n" source
  | _ ->
      String.concat
        (Printf.sprintf "    ( %S,\n      [\n" source
         :: List.map diagnostics ~f:(fun diagnostic ->
             Printf.sprintf "        %S;\n" (Refusal_control_scan.marker diagnostic))
        @ [ "      ] );\n" ])

let minus = Refusal_control_scan.minus

type row_difference = {
  absent : string list;  (** Extracted markers the row lacks. *)
  no_longer : string list;  (** Row markers no longer extracted. *)
  first_misplaced : (int * string * string) option;
      (** With neither side lacking anything, the first 1-based entry where the order differs, as
          (position, extracted, registered). *)
}

(** How a [registered] row differs from the [extracted] markers, [None] when it holds them in order.
*)
let row_difference ~registered ~extracted =
  if List.equal String.equal extracted registered then None
  else
    let absent = minus extracted registered and no_longer = minus registered extracted in
    let first_misplaced =
      if List.is_empty absent && List.is_empty no_longer then
        List.findi (List.zip_exn extracted registered) ~f:(fun _ (e, r) -> not (String.equal e r))
        |> Option.map ~f:(fun (position, (e, r)) -> (position + 1, e, r))
      else None
    in
    Some { absent; no_longer; first_misplaced }

(** Writes [difference] for [source]'s row (its key without the [test/operations/] prefix) on
    stderr, with a short summary and the replacement row built from [diagnostics] -- never into a
    golden: the row a check holds the manifest to stays the hand-pasted one. *)
let eprint_row_difference ~source diagnostics difference =
  eprintf
    "%s: refusal-control row differs from extraction, %d added and %d removed (not part of the \
     golden):\n"
    source (List.length difference.absent)
    (List.length difference.no_longer);
  List.iter difference.absent ~f:(eprintf "  extracted, absent from the row: %s\n");
  List.iter difference.no_longer ~f:(eprintf "  in the row, no longer extracted: %s\n");
  Option.iter difference.first_misplaced ~f:(fun (position, e, r) ->
      eprintf
        "  same markers in a different order; first difference at entry %d: extracted %s, the row \
         has %s\n"
        position e r);
  eprintf
    "  replace its `raw_entries` row in test/support/refusal_control_manifest.ml with the row \
     below -- tools/refusal-manifest-sync.sh does, re-keying the evidence and rewriting the \
     goldens a rewording reaches:\n\
     %s"
    (row ~source diagnostics)

(** The [raw_direct_evidence] keys (without the [test/operations/] prefix) of [source]'s direct
    failures beyond its [registered] row -- consumed as a multiset, since identical formats need a
    control apiece -- that no entry answers and that this process did not record through
    [observe_failure]. A scan whose control arm runs in a separate process records its observations
    there, so a key named here may still be covered by that arm: the list is what to check, not a
    verdict. Which control observes a failure is a judgment, so the entries stay hand-written. *)
let unevidenced_failures ~source ~registered diagnostics =
  List.fold diagnostics ~init:(registered, []) ~f:(fun (remaining, keys) diagnostic ->
      let marker = Refusal_control_scan.marker diagnostic in
      if List.mem remaining marker ~equal:String.equal then (minus remaining [ marker ], keys)
      else
        let key = failure_key ~source ~identity:diagnostic.Refusal_control_scan.identity in
        match diagnostic.Refusal_control_scan.kind with
        | Refusal_control_scan.Fail
          when not
                 (Hash_set.mem observed_failures key
                 || List.Assoc.mem direct_evidence key ~equal:String.equal
                 || List.Assoc.mem catalogue_only key ~equal:String.equal) ->
            ( remaining,
              Option.value (String.chop_prefix key ~prefix:"test/operations/") ~default:key :: keys
            )
        | Refusal_control_scan.Fail | Refusal_control_scan.Claim -> (remaining, keys))
  |> snd
  |> List.dedup_and_sort ~compare:String.compare

(** The [direct_evidence] and [catalogue_only] keys no current direct-failure diagnostic answers to:
    [diagnostics_of] gives a catalogued source's extracted diagnostics, [None] for a source outside
    the catalogue. A reworded failure format changes its identity, leaving its old key silently
    dead. *)
let stale_classifications ~diagnostics_of =
  List.filter_map (direct_evidence @ catalogue_only) ~f:(fun (key, _) ->
      let source, identity = String.rsplit2_exn key ~on:':' in
      let answered =
        Option.value_map (diagnostics_of source) ~default:false ~f:(fun diagnostics ->
            List.exists diagnostics ~f:(fun diagnostic ->
                String.equal diagnostic.Refusal_control_scan.identity identity
                &&
                match diagnostic.Refusal_control_scan.kind with
                | Refusal_control_scan.Fail -> true
                | Refusal_control_scan.Claim -> false))
      in
      if answered then None else Some key)

(** The keys classified both ways: answered by a control and catalogued as answered by none. *)
let doubly_classified =
  List.filter_map direct_evidence ~f:(fun (key, _) ->
      Option.some_if (List.Assoc.mem catalogue_only key ~equal:String.equal) key)

(** The elements written more than once, each named once, in table order (gh-ocannl-1840). Applied
    to direct-evidence pairs or catalogue-only keys: a duplicate adds nothing, so the audit refuses
    it as a copy-paste slip. Distinct claims under one direct-evidence key remain meaningful
    (gh-ocannl-1572). *)
let repeated_pairs entries =
  List.filter_mapi entries ~f:(fun i pair ->
      let earlier = List.take entries i in
      Option.some_if (List.count earlier ~f:(Poly.equal pair) = 1) pair)

(** The [raw_direct_evidence] pairs written more than once ({!repeated_pairs}). *)
let repeated_evidence = repeated_pairs direct_evidence

(** A catalogue-only key says no control answers for it; a second entry adds nothing whatever its
    reason. *)
let repeated_catalogue_keys entries = repeated_pairs (List.map entries ~f:fst)

type offer = { key : string; claim : string }
(** A run's offer for a direct failure its row lists with no [raw_direct_evidence] or
    [raw_catalogue_only] entry, that ran unanswered (gh-ocannl-1571): its [key] (without the
    [test/operations/] prefix) and a passed [claim] that answers for it ({!answering_claims}).
    {!print_section} writes one per answering claim on stderr ({!offer_lines}), and
    [tools/refusal-manifest-sync.sh]'s engine reads them back from the runs' logs ({!parse_offer})
    to print the entry ready to paste for each failure the audit would refuse -- printed by none of
    its goldens. Which claim answers is still the manifest's judgment, so an entry is offered, and
    written only when its reader accepts it by name ([--accept-offer], gh-ocannl-1840); and since a
    scan whose control runs in another process observes its refusals there, a run whose claims
    answer for nothing says nothing: what no run answers for is the engine's to name, from the
    goldens. *)

(** One stderr line per answering claim of [claims]. A label spanning lines cannot travel on one, so
    it is never offered: an entry naming it is a hand edit. *)
let offer_lines ~key claims =
  List.filter_map claims ~f:(fun claim ->
      Option.some_if
        (not (String.mem claim '\n'))
        (Refusal_section_heads.offer_head ^ key ^ Refusal_section_heads.offer_answered_by ^ claim))

(** The offer a log [line] carries, or [None] for any other line. *)
let parse_offer line =
  Option.bind
    (String.chop_prefix (String.rstrip line) ~prefix:Refusal_section_heads.offer_head)
    ~f:(fun rest ->
      Option.map (String.substr_index rest ~pattern:Refusal_section_heads.offer_answered_by)
        ~f:(fun at ->
          {
            key = String.prefix rest at;
            claim =
              String.drop_prefix rest (at + String.length Refusal_section_heads.offer_answered_by);
          }))

(** The section [print] ends a scan's golden with: its heading, then the marker of each of
    [diagnostics] that [expected] (the source's registered row) lists and this run exercised -- a
    claim's once Verdict recorded it passing, a direct failure's once its refusal ran -- and the
    catalogue-only ones after them. [source] is the key a direct failure's observations are filed
    under. Exposed apart from [print] so that a control can print a section for diagnostics it
    extracted itself, through the same printer: [verdict_teardown]'s transcripts, which
    [tools/test-mutation-run.sh] replays through [tools/mutation-run.sh], are of runs ending here
    (gh-ocannl-1364).

    The section ends on {!Refusal_section_heads.closing_row}, printed whatever came before it
    (gh-ocannl-1251, gh-ocannl-1318): every golden that prints a section pins it, so a run that
    printed it decided every marker above it. Without it a run cut inside the markers read the same
    as a complete run whose claim failed, and a run whose only shortfall was one silenced refusal --
    every claim passing, so no teardown line -- read as cut short.

    A marker of [expected] it does not print because a claim behind it failed ({!why_not_printed}),
    or because [diagnostics] no longer holds its diagnostic, gets a
    {!Refusal_section_heads.not_printed_head} line on stderr saying so; any other marker it does not
    print had nothing behind it run. So a golden diff short of a marker reads as a failed claim or
    as a refusal that never ran (gh-ocannl-1499), and a run with no failed check writes no such
    line. *)
let print_section ~source ~expected diagnostics =
  let passed_labels = ref (Verdict.passed_labels ()) in
  let not_printed marker reason =
    eprintf "%s%s -- %s\n" Refusal_section_heads.not_printed_head marker (Verdict.one_line reason)
  in
  let failed = ref (Verdict.failed_checks ()) in
  let why marker diagnostic =
    why_not_printed ~direct_evidence ~failed:!failed
      ~key:(failure_key ~source ~identity:diagnostic.Refusal_control_scan.identity)
      diagnostic
    |> Option.iter ~f:(fun (reason, remaining) ->
        failed := remaining;
        not_printed marker reason)
  in
  let catalogued = ref [] in
  let rivals =
    List.filter_map diagnostics ~f:(fun d ->
        match d.Refusal_control_scan.kind with
        | Refusal_control_scan.Fail -> Some d.Refusal_control_scan.format
        | Refusal_control_scan.Claim -> None)
  in
  printf "\nSynthetic controls: scanner refusal diagnostics exercised by this control golden:\n";
  diagnostics
  |> List.iter ~f:(fun diagnostic ->
      let marker = Refusal_control_scan.marker diagnostic in
      if List.mem expected marker ~equal:String.equal then
        match diagnostic.Refusal_control_scan.kind with
        | Refusal_control_scan.Claim -> (
            match claim_exercises !passed_labels diagnostic with
            | None -> why marker diagnostic
            | Some remaining ->
                passed_labels := remaining;
                printf "%s%s\n" Refusal_section_heads.marker_indent marker)
        | Refusal_control_scan.Fail -> (
            let key = failure_key ~source ~identity:diagnostic.Refusal_control_scan.identity in
            match
              standing ~direct_evidence ~catalogue_only
                ~observed:(Hash_set.mem observed_failures key)
                ~refused:(List.concat_map (Hashtbl.find_multi refused_runs source) ~f:refusal_lines)
                ~claim_lines:(claim_lines ~source) ~rivals ~passed_labels:(Verdict.passed_labels ())
                ~key diagnostic
            with
            | Exercised -> printf "%s%s\n" Refusal_section_heads.marker_indent marker
            | Catalogue_only -> catalogued := marker :: !catalogued
            | Executed_yet_catalogue_only ->
                printf "%s%s\n" Refusal_section_heads.marker_indent marker;
                Verdict.fail
                  (Printf.sprintf
                     "%s is listed in `raw_catalogue_only`, yet this run executed it -- move its \
                      key to `raw_direct_evidence`, or drop it if the control observes it itself"
                     key)
            | Unexercised ->
                why marker diagnostic;
                (* A direct failure no entry classifies: offer the claims that answer for it. *)
                if
                  not
                    (List.Assoc.mem direct_evidence key ~equal:String.equal
                    || List.Assoc.mem catalogue_only key ~equal:String.equal)
                then
                  let claims =
                    answering_claims ~passed_labels:(Verdict.passed_labels ())
                      ~claim_lines:(claim_lines ~source) ~rivals diagnostic
                  in
                  let key =
                    Option.value (String.chop_prefix key ~prefix:"test/operations/") ~default:key
                  in
                  List.iter (offer_lines ~key claims) ~f:(eprintf "%s\n")));
  List.iter
    (minus expected (List.map diagnostics ~f:Refusal_control_scan.marker))
    ~f:(fun marker -> not_printed marker "its diagnostic is no longer in the source");
  if not (List.is_empty !catalogued) then (
    printf
      "Catalogue-only: direct failures no control executes (reasons in `raw_catalogue_only`):\n";
    List.iter (List.rev !catalogued) ~f:(printf "%s%s\n" Refusal_section_heads.marker_indent));
  printf "%s\n" Refusal_section_heads.closing_row

let print source =
  let source_path =
    if Stdlib.Sys.file_exists source then source
    else
      let basename = Stdlib.Filename.basename source in
      if Stdlib.Sys.file_exists basename then basename
      else Stdlib.Filename.concat "test/operations" source
  in
  let normalized = String.substr_replace_all source ~pattern:"\\" ~with_:"/" in
  let source =
    match String.substr_index normalized ~pattern:"test/operations/" with
    | Some position -> String.drop_prefix normalized position
    | None ->
        let local = "test/operations/" ^ Stdlib.Filename.basename normalized in
        (* A bare or [./]-prefixed name is this directory's scan whether or not it has a row yet;
           the bootstrap below must print the key the census derives, not the spelling. *)
        if
          List.Assoc.mem entries local ~equal:String.equal
          || String.equal (Stdlib.Filename.dirname normalized) "."
        then local
        else normalized
  in
  let diagnostics = Refusal_control_scan.diagnostics (In_channel.read_all source_path) in
  (* Bootstrap: a scan with no row, or an empty one, gets its row written out on stderr -- never
     into the golden, and never into the manifest, so the census claim in env_var_deps still decides
     whether the row was added. A registered row with no diagnostics to list has nothing to say. *)
  let registered = List.Assoc.find entries source ~equal:String.equal in
  let expected = Option.value registered ~default:[] in
  let row_source =
    Option.value (String.chop_prefix source ~prefix:"test/operations/") ~default:source
  in
  if List.is_empty expected && not (List.is_empty diagnostics && Option.is_some registered) then (
    eprintf
      "%s has %s row in Refusal_control_manifest; add this row to `raw_entries` in \
       test/support/refusal_control_manifest.ml (not part of the golden):\n"
      source
      (if Option.is_some registered then "an empty" else "no");
    eprintf "%s" (row ~source:row_source diagnostics))
  else
    (* A stale row -- a diagnostic gained, lost or reworded -- gets the same treatment, again only
       on stderr: the catalogue still holds the manifest to the extraction, and a marker outside the
       row prints nothing below, so the census claim in env_var_deps still decides. *)
    Option.iter
      (row_difference ~registered:expected
         ~extracted:(List.map diagnostics ~f:Refusal_control_scan.marker))
      ~f:(fun difference ->
        eprint_row_difference ~source:row_source diagnostics difference;
        match unevidenced_failures ~source ~registered:expected diagnostics with
        | [] -> ()
        | keys ->
            eprintf
              "  direct failures new to the row with no `raw_direct_evidence` entry, none recorded \
               through `observe_failure` in this run: unless the scan's separate control run \
               records it, each needs an entry naming the control claim that observes it, under \
               the key below -- once the row lists it, a run offers the claims that answer for it, \
               and tools/refusal-manifest-sync.sh prints the entry ready to paste\n";
            List.iter keys ~f:(eprintf "    %S\n"));
  print_section ~source ~expected diagnostics
