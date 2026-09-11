(** The explicit bridge from mechanically extracted scanner diagnostics to their assigned permanent
    control suites. A new diagnostic is absent here by default; the catalogue test holds this
    manifest equal to the extractor and holds every entry against the assigned golden.

    This table is inventory, not evidence. [print] emits a claim marker only when Verdict recorded a
    successful execution of the claim's diagnostic format, and emits a direct-failure marker only
    when the exact negative-control line assigned below was observed or the caught branch recorded
    its format through [observe_failure]. The audit consumes marker occurrences as a multiset, so
    identical formats still need one exercised control apiece. Adding a row here therefore cannot
    make its own golden pass. *)

open Base
open Stdio

let raw_entries =
  [
    ("ll_test_ratchet.ml", []);
    ( "atomic_file_rename_scan.ml",
      [
        "[scanner-refusal:7191f5e19a3640926ad61aeb2d7a349e] source-generation input has";
        "[scanner-refusal:c045a29a4e2f18a646143f07d6ec923b] missing corpus section";
        "[scanner-refusal:b5717d7e9eb6f55e37e9f5f57d114fc1] present and nonempty";
        "[scanner-refusal:5b91d4ebbeebe59d5615dd99fe73edfd] unexpected corpus section";
        "[scanner-refusal:bccd646d58b5010b1fe4eee6abe7eb53] corpus arguments must";
        "[scanner-refusal:fb72b56980e9afc888059f90e9734599] arguments the rule's";
        "[scanner-refusal:20f449385ebca8a7a25b2d306c8e0f44] scan cannot vouch";
        "[scanner-refusal:77ea18b8b8167a5861a77a9a6bbe45bf] unrelated rename spelling";
        "[scanner-refusal:298ab0be4c86a0dabfaeeddc5e67bd29] contributes its derived";
        "[scanner-refusal:dfeaf03eef363cf0944237d1f754cb5a] generated-source negative control";
        "[scanner-refusal:16447aab0e906595f837aaa44096abbc] Atomic_file";
        "[scanner-refusal:698fae6b12a0d52a71f789bc962a882b] raw rename exemption";
        "[scanner-refusal:ed6060063668d409574488fcb7fc8897] exemption matches exactly";
      ] );
    ( "agent_notes_structure.ml",
      [
        "[scanner-refusal:bc4968cf04b3b7f3212afef877977199] handed the notes";
        "[scanner-refusal:2ac555c09ba6fe90fc6321fecba16cab] the notes' bullets";
        "[scanner-refusal:ec487246d0ff9ad951333a2c56d4a990] names is reported";
        "[scanner-refusal:ed552c5b3e09a3046aa9cce5257cc148] every finding carries";
        "[scanner-refusal:348bb1f1fa0be73fb4796a31a76abaad] each flagged bullet";
        "[scanner-refusal:ec62d0a7d728c7ab15a2c9b8608c6404] every exemption still";
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
      [ "[scanner-refusal:2fcd4d491da746e829dc7cec63905091] self-certifying and provenance" ] );
    ( "bin_smoke_membership_scan.ml",
      [ "[scanner-refusal:9446215c7b3c2270741bbb1cd8300733] executable exactly once" ] );
    ( "cache_dir_ignores.ml",
      [
        "[scanner-refusal:e2e9d19ffe2ad7688396847bd1ee643a] the repository-root gitignore";
        "[scanner-refusal:0f307832357ee7ca58b9798a86530dac] arguments the rule's";
        "[scanner-refusal:88e191145bdba435fd0fb303d486d684] gitignore no longer";
        "[scanner-refusal:3076fb53f1ae34b427f82aaa89404cb3] root gitignore pattern";
        "[scanner-refusal:e2fdd92481a66b85f6f54b5c68513edd] sources this check";
        "[scanner-refusal:cd6f72db4a56b656b92efd2c4a07945f] the cache directory";
        "[scanner-refusal:a8ea78ba60f371d22646868a36871617] argument";
        "[scanner-refusal:4168ef8491021497635783bb1f7e4232] no source reads";
        "[scanner-refusal:298f6bafdca761627eb0f4dfecddf6d3] the built-in default";
        "[scanner-refusal:50ad95ff66f64313a7ce58dadb8aeda9] the cache directory";
      ] );
    ( "codegen_text_inventory.ml",
      [
        "[scanner-refusal:83dfdd0395c402128515f27b2f3b1b1c] the exclusion for";
        "[scanner-refusal:b1a8055bff15f499f2f143975afca3d7] reaches the stale-exclusion";
        "[scanner-refusal:123bedafe19167d3536ddb2ca1115d21] cannot say whether";
        "[scanner-refusal:532d3ed986dd6559fec87d780769ee1f] declares module";
        "[scanner-refusal:a59159ac0fd0f431da9519f8016e6762] every scanned root";
        "[scanner-refusal:c995ea0a058a228bf654449befcccb4b] meets its source-site";
        "[scanner-refusal:d246f03ca5c483db40b94796658cf426] every source handed";
        "[scanner-refusal:ebeeadb92d10e201894cb4b9fb9021bb] every exclusion still";
        "[scanner-refusal:545694a8ea6e371db823bcc7d2639f85] scanned library interfaces";
        "[scanner-refusal:327f5dea331c9591f8d59a6da296202d] scanned library declares";
        "[scanner-refusal:80b2a504ebea9c64f6fcc3c5da12164e] generated text behind";
      ] );
    ( "config_dep_completeness.ml",
      [
        "[scanner-refusal:2ade05f14bd972f85a1792722f8a0ab6] arguments the rule's";
        "[scanner-refusal:6c29eb9b1db92bba229707a7cd8e1b5b] Dune_stanza_scan.action_heads";
        "[scanner-refusal:cebf445823df462f70b459a86b20e3d3] Dune_stanza_scan";
        "[scanner-refusal:6d151bed80a45201e61d096c2c38d210] which has no";
        "[scanner-refusal:5509acc7fac8ff0412e3631b49c06dc7] change directory without";
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
        "[scanner-refusal:7f7db667dc63d73faefa570451cf5f2b] ambiguous command-line value";
        "[scanner-refusal:7a3d690e9ef5541a1c754952c6cf6877] ambiguous bare config";
        "[scanner-refusal:200477f5fecf3e08ff5055af77a00298] prefix-free config flag";
        "[scanner-refusal:8acb2b9087cf4a7e4112382c5e058e1a] Utils.known_config_keys";
        "[scanner-refusal:085d7b5a19f012e46728008a42120c74] non-config environment exemptions";
        "[scanner-refusal:5d123838dd21fb3fd4a7529e51b8a74f] non-config environment-mention \
         occurrence";
        "[scanner-refusal:3252eebc4149f3005426dbd4385702c1] non-config assignment exemptions";
        "[scanner-refusal:4f2e0da970aa70c6ff3549e9e435561f] non-config assignment exemption";
        "[scanner-refusal:2908ea301f2b18efa6c95e6ab0302abf] historical invalid-config exemption";
        "[scanner-refusal:6d76bc1e86b8d630263d70bd7c30e50b] config-mention occurrence counts";
        "[scanner-refusal:8c14e661ffe9cdf859b5111394d557a8] prefix-free config-mention occurrence";
        "[scanner-refusal:7c5c4cc0bd161e76fe99e9983d2db70f] command-line value occurrence";
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
        "[scanner-refusal:84bf7dbb3e9182b1eced5e76a271e5f6] config-usage direct refusal";
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
        "[scanner-refusal:ab97825eb17bf92bf92973b99e0849ff] mli-less implementation modules";
        "[scanner-refusal:20f449385ebca8a7a25b2d306c8e0f44] scan cannot vouch";
        "[scanner-refusal:085de8c521f251df70b65fc6454dfdad] OCaml source does";
        "[scanner-refusal:e8c85fd2a3138c8fe3283ab67595826c] zero-reference implicit export";
        "[scanner-refusal:8e094223fc3e55912db0c3f53c9fcaca] named dead-export exemptions";
        "[scanner-refusal:e32dfff95d976581bde669214a7e6415] dead-export exemption remains";
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
        "[scanner-refusal:55743888dd2a4a2234ee1d3f031ea47e] meets its source-count";
      ] );
    ( "env_var_deps.ml",
      [
        "[scanner-refusal:2b65edf20e36d983c1a016b72c2aadd1] plugged-in check receives";
        "[scanner-refusal:35bab4283cd8b87d51ca3476cb0e971e] source_tree";
        "[scanner-refusal:51b4c42723bc3fa6bfc678d091151ed9] source_tree";
        "[scanner-refusal:3182e84964718fd8189a6dd0d93d14b1] diagnostic absent from";
        "[scanner-refusal:8daa7afe8b321ab7932dac6060588501] same diagnostic fragment";
        "[scanner-refusal:08ac167f3e68d870e2cfb7f7c85accc3] control marker occurrence";
        "[scanner-refusal:e68d9277c8a3a5c061164578baf2f632] the population equality";
        "[scanner-refusal:87659b49df04eee3fc19350b0be6629a] repo-relative scanner paths";
        "[scanner-refusal:490ac5e3405d328c2953c38df4f0974b] arguments the rule's";
        "[scanner-refusal:1356c3ade8d75f4aeb6332c69e239db3] the resource-lifecycle instrumentation";
        "[scanner-refusal:09e01a62cce1d340393a2244ed33ea10] Test_utils.Generated";
        "[scanner-refusal:034280d144ee4338406ff5f6a1cf287d] comment that does";
        "[scanner-refusal:30c88c1631600d7969a9aefd4de74162] carries a backend";
        "[scanner-refusal:f2633aa4ebd6f2c33fd21e4d69cc9786] one backend marker";
        "[scanner-refusal:2803edcc7de84ea6d928343b9d8c8ad2] env_var";
        "[scanner-refusal:533fc9999ad0841992e0c1734d9c1f5d] env_var";
        "[scanner-refusal:7336dd5a712b0c3a578b64d7defb1654] between stanzas declares";
        "[scanner-refusal:1aaab6e3cd8f4615e1cd1d0321e309a1] times and only";
        "[scanner-refusal:f11c09a8318b3a7863444256973f57d4] running an executable";
        "[scanner-refusal:95919586e2a2b25eaaf13ca8093ab584] repository-wide scan rule";
        "[scanner-refusal:be41f253ce4b808b6b4ec587c304ec08] repository-wide scan rule";
        "[scanner-refusal:b040632469771b25abb3469878413be3] the repository scanner";
        "[scanner-refusal:dda5b4655f74eda46ec0c60e320d232b] run's generated artifacts";
        "[scanner-refusal:747372ab4b28cca8978c2b6bd8983005] env_var";
        "[scanner-refusal:364d32b22a5cb7a70b569db0d8864170] env_var";
        "[scanner-refusal:07e7987f0d61b10a9fea6c6590c73d59] env_var";
        "[scanner-refusal:bbadfd8513559355fedbc54dc3dc678a] serializes its actions";
        "[scanner-refusal:eb5d5419b3bd18240469743cbabc58f2] has actions on";
        "[scanner-refusal:a06dccc27afe2153a7700609691d3c8d] attaches a rule";
        "[scanner-refusal:cd2777d83ada3318386d3cd02b39018b] repository a repo-wide";
        "[scanner-refusal:ffdd3e3b7a0db8da177f376b13f03c45] repository to produce";
        "[scanner-refusal:f16784d0b1f52de5e30378cb771ada5c] alias does not";
        "[scanner-refusal:4012b24c7623acd977522e8f2eacf2bb] attaches a rule";
        "[scanner-refusal:ba5d90fc4bcec944ebd169329c706968] goldens on the";
        "[scanner-refusal:c1a398b587b5de3d7d02145dc45274a6] alias <suite>-<name> that";
        "[scanner-refusal:941a9fb77d867c7b1576a0f19ee5db3b] declares";
        "[scanner-refusal:2b67d02d52c3465bfba40d167802f0a7] env_var";
        "[scanner-refusal:0deef0cb94c45615f32ee84ac2e76f8e] declares tracing gates";
        "[scanner-refusal:02e183a4105132ec092206c1d35d6247] reads the tracing";
        "[scanner-refusal:2df546d04b63387a24409b3c4c6a9611] declares the tracing";
        "[scanner-refusal:e55f77a62871fb685d23199e95c8f173] the environment variable";
        "[scanner-refusal:d5f94657407bc4dbf949ba4187e80100] include_subdirs";
        "[scanner-refusal:ea6e2adedc7afd62fa55afcf859f251d] names the module";
        "[scanner-refusal:55f2feade4e139b5abb073d0b89c42cb] reaches the environment";
        "[scanner-refusal:267a42b2812f4679c4463495cc199d36] reads the configuration";
        "[scanner-refusal:41d9bc729b5b8b3908a48d45b1e0203c] reads the configuration";
        "[scanner-refusal:a323f01100bfd685e76c62154cf4eaae] reads the configuration";
        "[scanner-refusal:e45e166af22d229414296459a3ca022a] scanner refusal";
        "[scanner-refusal:fe4c53f899eb08a9ae67e7a797a5a841] scanner-refusal exemptions no";
        "[scanner-refusal:0b33370125e32eaa4bb0f50c6cc3bc1a] exempted declarations no";
        "[scanner-refusal:638620a0745d0a136810c5b0256e0ec0] directories exempted from";
        "[scanner-refusal:7d1923374b5bbf7028b59f2670772e8c] executable either declares";
        "[scanner-refusal:68816c7803a2eabaa2a2d8799ba00a78] running an executable";
        "[scanner-refusal:a90efecb7feb61485c0886565a0e4ba9] running an executable";
        "[scanner-refusal:09ce44850cdf45c95e308d6c64aed813] the repository's";
        "[scanner-refusal:1bb3be04ec494f4917d1537675671f1d] repository-wide derivation finds";
        "[scanner-refusal:b35f6c5cf70c9d942c2e42a4c1af9975] Test_utils.Generated.init";
        "[scanner-refusal:9ad60261dbb2540d4d7e060be725ad4d] repository's census finds";
        "[scanner-refusal:b54873c01dcc148c1691a882e4181993] Test_utils.Generated.init";
        "[scanner-refusal:7ff972d0b76d472194796b39466092aa] Test_utils.Generated.init";
        "[scanner-refusal:f8d375e9267abafa87ce8eec2d53731d] repository-wide scan rules";
        "[scanner-refusal:f1c1e39705142c7bdc92b3f0ede72582] refusal-control manifest source";
        "[scanner-refusal:1f42d7a2739170c96004439d524c5d96] permanent control-golden corpus";
        "[scanner-refusal:a2ef9338814a08816bb85a6bab6ac71c] statically recoverable scanner";
        "[scanner-refusal:96e2132b9ea14b4c19e7573b7725b9ec] omits the declaration";
        "[scanner-refusal:2eb428f8dfd55908e0cd2236bc4505b4] declaration added passes";
        "[scanner-refusal:055900a8a1fbefd29b7c64a46953700f] external command handed";
        "[scanner-refusal:fac924c455e2114b6e6bdebdc44811fd] same stanza declaring";
        "[scanner-refusal:b4a6c6ad5ae1c0d26b7fe78da5c5aa7f] command handed nothing";
        "[scanner-refusal:a6ad129bb27775f8c16abc37c6c39ed6] metal-codegen alias does";
        "[scanner-refusal:d2a4a4c55f9409ff2a808631d0753f27] family stanza passes";
        "[scanner-refusal:8ebeaee3c8fee72d0c6d5b1df2fa0c34] the resource-lifecycle instrumentation";
        "[scanner-refusal:da4ac19271accffacfcd9a6bbe210e22] member listed passes";
        "[scanner-refusal:3f9f68401bac53be0cbb38ce715e460d] executable whose RUNNER";
        "[scanner-refusal:ed1dfb1acf49ea77d9b2bdd8ad92d312] alias listed passes";
        "[scanner-refusal:ddd1defde5b6df4dab1d802e19cd88ef] generated aliases listed";
        "[scanner-refusal:46c759bca27349aa081cf5efe33bdef0] same plural stanza";
        "[scanner-refusal:aff85d14fb92bfaa9802e45b964c7292] reads the instrumentation";
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
        "[scanner-refusal:8043b9ea16535c8062dcc52282cdb90a] through another qualifier";
        "[scanner-refusal:1cb2feb54940b69b22887119d96e5203] Alloc_census.snapshot";
        "[scanner-refusal:3c9662bb21fae53a6e850ec0b1d533fa] I.Alloc_census.snapshot";
        "[scanner-refusal:aa4e4d279169f4e2990023a88a8a29fc] Alloc_census";
        "[scanner-refusal:de5cf6819a64d46d5a27eb8c300af31b] Ir.Alloc_census";
        "[scanner-refusal:84e456282a1ff74968abe67ae2fa4694] executable's public name";
        "[scanner-refusal:1e238caa905efec1f7ae0cf764033e25] Vendor.Ir";
        "[scanner-refusal:00e63d0fbcdd91be8bea4b0244c7860e] Vendor.Ir.Alloc_census";
        "[scanner-refusal:2a67fb33bc312d6cfdc1ce0f15d513bd] Foo.Alloc_census";
        "[scanner-refusal:f84656abac178f5d2b2d79677353a343] Vendor.I.Alloc_census";
        "[scanner-refusal:f63c79247666a73ea1dc618899111b77] functor parameter named";
        "[scanner-refusal:5e4f4234263c64ddf6bc2236f1b4032c] I.Alloc_census";
        "[scanner-refusal:c4b1b7a8d9409c62e651fec93e8bb1a3] structure defines something";
        "[scanner-refusal:2d6a378a5a45d9bb99e928532c0226b7] somebody else re-exports";
        "[scanner-refusal:b4b3077aecacdd13f81ef68627f7e8ff] functor's parameter named";
        "[scanner-refusal:e34dc9c194473eab7a73109597d033c0] includes the qualifier";
        "[scanner-refusal:081886dc55c593d16ce4e598ecc604c1] Ir.Alloc_census.t";
        "[scanner-refusal:5940ea2811e203c19adf2bb49003d448] definition supersedes it";
        "[scanner-refusal:15b55a16ec2a2d4b14898f888710526f] Alloc_census";
        "[scanner-refusal:73cf957dbac466cdcb1ed757fcff1aa1] qualifier's name shadows";
        "[scanner-refusal:b2ffc9450c3e952a03abd091b5abea61] later reference alive";
        "[scanner-refusal:bb4952563f8b6d5b4e642fe1130df293] alias declared before";
        "[scanner-refusal:6798c11e122a65375ac4a8b22ddfd20b] includes the qualifier";
        "[scanner-refusal:8f7f708a45a89228e1e478f3ddb158ae] I.Alloc_census.t";
        "[scanner-refusal:be7f67353f6c0f8782c3b867b1023af5] runtest-<name> dune generates";
        "[scanner-refusal:dc2fb82ae5ada84e736e180065bc2881] reads the instrumentation";
        "[scanner-refusal:27dfb3ff299833db9842614f2e2b5384] local executable's name";
        "[scanner-refusal:0880ce931733097b9e44c86e1f0b7065] ANOTHER directory's binary";
        "[scanner-refusal:93ed6fc086891252db10d64e8dbfeca0] alias dune generates";
        "[scanner-refusal:84a1eda5072c0c3e43ecad5c2af6e561] stanza neither derivation";
        "[scanner-refusal:e38848b2fb4d8a90933cfe88e13cc04e] guard neither declares";
        "[scanner-refusal:2eb428f8dfd55908e0cd2236bc4505b4] declaration added passes";
        "[scanner-refusal:605b6d65216f3c4fba7184a09d392f05] pinning the variable";
        "[scanner-refusal:351acc91087a8ad372f2b9d40f2c1d20] SIBLING branch of";
        "[scanner-refusal:c0e7be9911f1a357c38e840ace916450] pinned guard does";
        "[scanner-refusal:c417aa0905f775b98ac3a197ecdd454c] dune's default module";
        "[scanner-refusal:bf505697938fab0cadb9434dc171764b] program declared inside";
        "[scanner-refusal:9b0ca5fe801d10403fb73978e58049f1] module the stanza";
        "[scanner-refusal:3aaf1e15d8b245e1193c7d00698c8d84] utils.ml";
        "[scanner-refusal:ba0a6cca6e0d6cef1cf5e8661a56b5a0] reported as undeclarable";
        "[scanner-refusal:21fd01e1a9e76489f599ef5a9387c17f] and pinning it";
        "[scanner-refusal:23b8372cc11b73d08a28c880b778eaf2] itself pins through";
        "[scanner-refusal:67415c1f44643b671427afd9352d8ba4] refused not approximated";
        "[scanner-refusal:422d713c4296fbd0c9cc8cace51b682d] same directive inside";
        "[scanner-refusal:a317ce82c0267d509ae45cf1d1b349d7] reported there being";
        "[scanner-refusal:c0f203aa0cf72d22c483afaf430d5353] library with inline";
        "[scanner-refusal:afc8337bcb6f3d267d65774a59ca7f86] inline_tests";
        "[scanner-refusal:b850c5e69abfcc5facf9b920af1a291f] same executable answers";
        "[scanner-refusal:b098b95e7bbcc632b19f0e470447fd97] both declaring passes";
        "[scanner-refusal:d8c7b4408936ac9480281388a26eb873] dynamic reach whose";
        "[scanner-refusal:7a38e9cb5b4418e6e5bb8fe3b294d8fd] the same variable";
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
    ( "optional_arg_inventory.ml",
      [
        "[scanner-refusal:924efa1e307f7c5a40712dd66311abe1] the inventory nothing";
        "[scanner-refusal:52e64fbae740ca90c4066b7840e926ae] source hand-over reaches";
        "[scanner-refusal:8827ebeb8572d08d2dd427695c19c0f4] lib/ optional arguments";
        "[scanner-refusal:69241f287bacdd3e1e418debbdc13d3d] every unimplemented option";
      ] );
    ( "shell_scripts_parse.ml",
      [
        "[scanner-refusal:88b12da9841dbf6131676bf4fb0f4686] statement-position ! command";
        "[scanner-refusal:003eba7381c87352fb02cd374fa62ea9] errexit-negation fixture";
        "[scanner-refusal:b67c631ab7ecd7df55fc002761cf99f6] reaches the absent()-style";
        "[scanner-refusal:ac074b168e19f7f99423326fcdea5669] shebang";
        "[scanner-refusal:8743bc48ff2a3da0c15e40f654ad6ff7] shebang";
        "[scanner-refusal:15ee2c44c18fee32abb92315075e99ab] parses";
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
        "[scanner-refusal:55743888dd2a4a2234ee1d3f031ea47e] meets its source-count";
      ] );
    ( "verdict_ratchet.ml",
      [
        "[scanner-refusal:977473ba169d481e2b57377845e9a8b8] exempted quantified bindings";
        "[scanner-refusal:52b0c212a9f1ba97a1400cd478d7cfda] shadowing definitions separate";
        "[scanner-refusal:28926aa707bb0e8c7e5a087d40f0bf13] exempted claim-shaped literal";
        "[scanner-refusal:0f307832357ee7ca58b9798a86530dac] arguments the rule's";
        "[scanner-refusal:f3e306858423827e6575d13a17d57af6] check cannot vouch";
        "[scanner-refusal:53b71ea33a69d24c941985f4eb35406b] prints the claim";
        "[scanner-refusal:7d1962ff34bbe1ed7b89a535dc0cfd7e] exempted literals that";
        "[scanner-refusal:ad023891a3bd7827b3905292c487055b] claim-shaped literal any";
        "[scanner-refusal:e1354d390a356f0c79420c25232b0b42] planted canaries the";
        "[scanner-refusal:b3bbed986e758647a2db67f9428972c6] claims through Verdict";
        "[scanner-refusal:63d9475f3df44af6baac8bdc03a3310b] every quantified binding";
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
      ] );
  ]

let entries =
  List.map raw_entries ~f:(fun (source, markers) -> ("test/operations/" ^ source, markers))

let markers source = List.Assoc.find_exn entries source ~equal:String.equal
let sources = List.map entries ~f:fst

let raw_direct_evidence =
  [
    ( "codegen_text_inventory.ml:123bedafe19167d3536ddb2ca1115d21",
      "ok: source -- text the scan cannot name marks the itemisation partial, without losing the \
       file" );
    ( "codegen_text_inventory.ml:532d3ed986dd6559fec87d780769ee1f",
      "ok: rejection -- opening the emitter's module hides the render" );
    ( "config_dep_completeness.ml:2ade05f14bd972f85a1792722f8a0ab6",
      "ok: a program action's arguments are not actions" );
    ( "config_dep_completeness.ml:6c29eb9b1db92bba229707a7cd8e1b5b",
      "ok: a rule that copies an executable does not run it" );
    ( "config_dep_completeness.ml:cebf445823df462f70b459a86b20e3d3",
      "ok: raw stanzas -- a name the stanza binds resolves under one too" );
    ( "config_dep_completeness.ml:6d151bed80a45201e61d096c2c38d210",
      "ok: copies the config -- no copy_files at all" );
    ( "config_dep_completeness.ml:5509acc7fac8ff0412e3631b49c06dc7",
      "ok: a test's shell action leaves its directory unestablished too" );
    ( "config_dep_completeness.ml:e11f1dd0ebe8035f6b6bb919c904e611",
      "ok: a named dep may wrap its path in a dependency form" );
    ( "config_dep_completeness.ml:e20c1b4ad3881b8147939aef02357ab2",
      "ok: an action head on neither list is reported" );
    ( "config_dep_completeness.ml:332abda7feebdaa6c1c10670865ea5c6",
      "ok: a library's own deps are not the inline tests' deps" );
    ( "config_dep_completeness.ml:c35e7bf895b2bfc334caa198f2db4c83",
      "ok: raw stanzas -- a tool only reading them is the same text" );
    ( "config_dep_completeness.ml:a83ec7419ab530468bb3d4143c8c8576",
      "ok: raw stanzas -- a tool only reading them is the same text" );
    ( "config_dep_completeness.ml:28c35655d1568c3c11bdc1893d8847fa",
      "ok: raw stanzas -- a bare command under setenv PATH is unnameable" );
    ( "config_dep_completeness.ml:929a18d24ebaad373970a360298e3b88",
      "ok: raw stanzas -- a tool only reading them is the same text" );
    ( "config_dep_completeness.ml:328553bf93b184a97ab7d44d97c369a4",
      "ok: raw stanzas -- a library's preprocessor is not a test-running rule" );
    ( "digest_completeness.ml:7cab0a5763f4811f331a4aab327c6784",
      "ok: key list -- a list written at the iteration" );
    ( "digest_completeness.ml:fc3ef3788212f2d1f5a9d9a46be3ca38",
      "ok: predicate call contributes its keys -- with_runtime_debug" );
    ( "digest_completeness.ml:dc9995a5d5b80370dc90971d3fd019df",
      "ok: predicate call contributes its keys -- with_runtime_debug" );
    ( "digest_completeness.ml:a385ae718c8af3ab39a45701b75fda4d",
      "ok: predicate call contributes its keys -- with_runtime_debug" );
    ( "digest_completeness.ml:14b02d2180d3814d6746acd827aecbc3",
      "ok: Generated.init -- a bare init without the open is somebody else's function" );
    ( "digest_completeness.ml:8109160f59717560ff5e6f7251a6a2b2",
      "ok: an escape sequence decodes to the real key" );
    ( "digest_completeness.ml:1825a277ba0eda3313fbe079e465c15d",
      "ok: environment read -- a function of the file's own that happens to share the name is not \
       the reader" );
    ( "env_var_deps.ml:490ac5e3405d328c2953c38df4f0974b",
      "a rule reusing the alias dune generates for a `(test)` in the same `(subdir …)` group is \
       reported there too: true" );
    ( "env_var_deps.ml:1356c3ade8d75f4aeb6332c69e239db3",
      "a source that names the instrumentation in a comment, a string and a longer identifier \
       reads none of it, and is no member: true" );
    ("env_var_deps.ml:09e01a62cce1d340393a2244ed33ea10", "derivation calls a member.");
    ( "env_var_deps.ml:034280d144ee4338406ff5f6a1cf287d",
      "a source that names the instrumentation in a comment, a string and a longer identifier \
       reads none of it, and is no member: true" );
    ( "env_var_deps.ml:30c88c1631600d7969a9aefd4de74162",
      "an executable declared in a `(subdir …)` group is aggregated by the top-level family stanza \
       when the rule that runs it sits at the top level: true" );
    ( "env_var_deps.ml:f2633aa4ebd6f2c33fd21e4d69cc9786",
      "a stanza whose backend marker names metal is reported, naming its family, when the \
       metal-codegen alias does not reach it: true" );
    ( "env_var_deps.ml:2803edcc7de84ea6d928343b9d8c8ad2",
      "a stanza whose backend marker names metal is reported, naming its family, when the \
       metal-codegen alias does not reach it: true" );
    ( "env_var_deps.ml:533fc9999ad0841992e0c1734d9c1f5d",
      "an executable run by a `(test)` stanza's custom action is aggregated through the \
       `runtest-<name>` dune generates for that test: true" );
    ( "env_var_deps.ml:7336dd5a712b0c3a578b64d7defb1654",
      "a stanza whose backend marker names metal is reported, naming its family, when the \
       metal-codegen alias does not reach it: true" );
    ( "env_var_deps.ml:1aaab6e3cd8f4615e1cd1d0321e309a1",
      "a dune file whose module sets this scan cannot place is refused, not approximated: true" );
    ( "env_var_deps.ml:f11c09a8318b3a7863444256973f57d4",
      "a rule running a file that shares the executable's public name is not its runner, however \
       alike the two strings are: true" );
    ( "env_var_deps.ml:95919586e2a2b25eaaf13ca8093ab584",
      "a runner that runs another directory's executable of the same name does not aggregate this \
       directory's member: true" );
    ( "env_var_deps.ml:be41f253ce4b808b6b4ec587c304ec08",
      "a runner that runs another directory's executable of the same name does not aggregate this \
       directory's member: true" );
    ( "env_var_deps.ml:b040632469771b25abb3469878413be3",
      "a module the stanza names and this check was handed no source for is reported, not read as \
       one that makes no reads: true" );
    ( "env_var_deps.ml:dda5b4655f74eda46ec0c60e320d232b",
      "an executable run by a `(test)` stanza's custom action is aggregated through the \
       `runtest-<name>` dune generates for that test: true" );
    ( "env_var_deps.ml:747372ab4b28cca8978c2b6bd8983005",
      "nor does writing `Vendor.Ir.Alloc_census` out in full: a path names the module it starts \
       at: true" );
    ( "env_var_deps.ml:364d32b22a5cb7a70b569db0d8864170",
      "a runner that runs another directory's executable of the same name does not aggregate this \
       directory's member: true" );
    ( "env_var_deps.ml:07e7987f0d61b10a9fea6c6590c73d59",
      "a test directory's own `utils.ml` is not the module that defines the reader, and is not \
       exempt: true" );
    ( "env_var_deps.ml:bbadfd8513559355fedbc54dc3dc678a",
      "a `(subdir …)` group whose actions take the training lock and whose gate does not is \
       reported there too: true" );
    ( "env_var_deps.ml:eb5d5419b3bd18240469743cbabc58f2",
      "a rule on a `(test)`'s generated alias that runs ANOTHER directory's binary is not the \
       deliberate gate collision, and is reported: true" );
    ( "env_var_deps.ml:a06dccc27afe2153a7700609691d3c8d",
      "a public name belongs to its own executable, so a rule running the second unit's public \
       name does not aggregate the first: true" );
    ( "env_var_deps.ml:cd2777d83ada3318386d3cd02b39018b",
      "a public name belongs to its own executable, so a rule running the second unit's public \
       name does not aggregate the first: true" );
    ( "env_var_deps.ml:ffdd3e3b7a0db8da177f376b13f03c45",
      "an executable run by a `(test)` stanza's custom action is aggregated through the \
       `runtest-<name>` dune generates for that test: true" );
    ( "env_var_deps.ml:f16784d0b1f52de5e30378cb771ada5c",
      "a runner that runs another directory's executable of the same name does not aggregate this \
       directory's member: true" );
    ( "env_var_deps.ml:4012b24c7623acd977522e8f2eacf2bb",
      "an executable run by a `(test)` stanza's custom action is aggregated through the \
       `runtest-<name>` dune generates for that test: true" );
    ( "env_var_deps.ml:ba5d90fc4bcec944ebd169329c706968",
      "the same tree with the runner's own alias listed passes: true" );
    ( "env_var_deps.ml:c1a398b587b5de3d7d02145dc45274a6",
      "an executable run by a `(test)` stanza's custom action is aggregated through the \
       `runtest-<name>` dune generates for that test: true" );
    ( "env_var_deps.ml:941a9fb77d867c7b1576a0f19ee5db3b",
      "a guard in a plain library is reported, there being no `deps` field in reach to declare it: \
       true" );
    ( "env_var_deps.ml:2b67d02d52c3465bfba40d167802f0a7",
      "`(env_var OCANNL_BUILD_FILES_PREFIX)`. Nothing else differs between the two runs." );
    ( "env_var_deps.ml:0deef0cb94c45615f32ee84ac2e76f8e",
      "a module the stanza names and this check was handed no source for is reported, not read as \
       one that makes no reads: true" );
    ( "env_var_deps.ml:02e183a4105132ec092206c1d35d6247",
      "a family alias defined inside a `(subdir …)` group needs that group's own ambient gate, and \
       is reported without one: true" );
    ( "env_var_deps.ml:2df546d04b63387a24409b3c4c6a9611",
      "two stanzas that omit `(modules …)` get a main each, so only the one whose main reads the \
       instrumentation is a member: true" );
    ( "env_var_deps.ml:e55f77a62871fb685d23199e95c8f173",
      "an executable run by a `(test)` stanza's custom action is aggregated through the \
       `runtest-<name>` dune generates for that test: true" );
    ( "env_var_deps.ml:d5f94657407bc4dbf949ba4187e80100",
      "a dune file whose module sets this scan cannot place is refused, not approximated: true" );
    ( "env_var_deps.ml:ea6e2adedc7afd62fa55afcf859f251d",
      "a module the stanza names and this check was handed no source for is reported, not read as \
       one that makes no reads: true" );
    ( "env_var_deps.ml:55f2feade4e139b5abb073d0b89c42cb",
      "the key list is resolvable at all. Nothing else differs between the runs." );
    ( "env_var_deps.ml:267a42b2812f4679c4463495cc199d36",
      "and declaring the variable in `(inline_tests (deps …))` is no licence: that invalidates the \
       inline runner alone: true" );
    ( "env_var_deps.ml:41d9bc729b5b8b3908a48d45b1e0203c",
      "The guard rule is put to a tree of one `(executable)` whose module reads a configuration key"
    );
    ( "env_var_deps.ml:a323f01100bfd685e76c62154cf4eaae",
      "the checker reports the key and exits 1 when the rule running the guard neither declares \
       nor pins it: true" );
    ( "env_var_deps.ml:e45e166af22d229414296459a3ca022a",
      "appears in no permanent control golden in the negative arm, and appears in the positive arm."
    );
    ( "env_var_deps.ml:fe4c53f899eb08a9ae67e7a797a5a841",
      "a family alias defined inside a `(subdir …)` group needs that group's own ambient gate, and \
       is reported without one: true" );
    ( "env_var_deps.ml:0b33370125e32eaa4bb0f50c6cc3bc1a",
      "`open Vendor.Ir` and `module Ir = Vendor.Ir` bind Vendor's module, not the qualifier, so \
       neither makes their file a member: true" );
    ( "env_var_deps.ml:638620a0745d0a136810c5b0256e0ec0",
      "a family alias defined inside a `(subdir …)` group needs that group's own ambient gate, and \
       is reported without one: true" );
    ( "env_var_deps.ml:09ce44850cdf45c95e308d6c64aed813",
      "a stanza neither derivation calls a member is asked for no family alias: true" );
    ( "env_var_deps.ml:b35f6c5cf70c9d942c2e42a4c1af9975",
      "an external command handed a file this workspace builds is a stanza the rule reaches, \
       reported by name when it declares neither: true" );
    ( "env_var_deps.ml:9ad60261dbb2540d4d7e060be725ad4d",
      "a dynamic reach whose keys resolve to nothing is refused rather than passed over in \
       silence: true" );
    ( "test_config_consistency.ml:40f6a4df02a115c327391e051bacd5c6",
      "ok: predicate call contributes its keys -- with_runtime_debug" );
    ( "test_config_consistency.ml:dfd9b6901bf188f6500bcc1284778dcd",
      "ok: predicate call contributes its keys -- with_runtime_debug" );
    ( "test_config_consistency.ml:a22f67563f6e58a972758cfb6438b236",
      "ok: environment read -- a function of the file's own that happens to share the name is not \
       the reader" );
    ( "test_config_consistency.ml:f183d4fa8451402727499389f65d55a4",
      "ok: environment read -- a function of the file's own that happens to share the name is not \
       the reader" );
    ( "test_config_consistency.ml:9a0c38727d642eb678b4707766f25708",
      "ok: predicate call contributes its keys -- with_runtime_debug" );
    ( "test_config_consistency.ml:1ce67444f994f72961fbc476c23dcd84",
      "ok: predicate call contributes its keys -- with_runtime_debug" );
    ( "test_config_consistency.ml:2651df7744050e9d7f649f390b7c53b1",
      "ok: a record literal is not a quoted string" );
    ( "test_config_consistency.ml:3a6832937329cda23c716c3883e04cb2",
      "ok: Generated.init -- a bare init without the open is somebody else's function" );
    ( "test_config_consistency.ml:8bb76ad7cf948a8181b88ee36b0148c6",
      "ok: key list -- a list written at the iteration" );
    ( "test_config_consistency.ml:11c8ffb35753ffb6f87da20d2ba6654f",
      "ok: settings read -- an unqualified record of the same shape is not a read" );
    ( "test_config_consistency.ml:c56af001748a614d221709bf11098511",
      "ok: an empty literal names no key, so no key is read" );
    ( "test_config_consistency.ml:e583c416783b3632963a66636f20f65e",
      "ok: a record literal is not a quoted string" );
    ( "test_config_consistency.ml:b4c59d5ad7a9c4e591998fccf8f039aa",
      "ok: key list -- a list written at the iteration" );
    ( "verdict_ratchet.ml:0f307832357ee7ca58b9798a86530dac",
      "ok: not a claim -- nothing before it but a blank line" );
    ("verdict_ratchet.ml:f3e306858423827e6575d13a17d57af6", "ok: claim shape -- the plain form");
    ( "verdict_ratchet.ml:53b71ea33a69d24c941985f4eb35406b",
      "ok: not a claim -- nothing before it but a blank line" );
    ( "verdict_ratchet.ml:7d1962ff34bbe1ed7b89a535dc0cfd7e",
      "ok: source -- a claim inside a list is reached" );
    ( "verdict_ratchet.ml:ad023891a3bd7827b3905292c487055b",
      "the walk counts every string literal it passes: true" );
    ("verdict_ratchet.ml:e1354d390a356f0c79420c25232b0b42", "ok: claim shape -- the planted canary");
  ]

let direct_evidence =
  List.map raw_direct_evidence ~f:(fun (key, evidence) -> ("test/operations/" ^ key, evidence))

let observed_failures = Hash_set.create (module String)
let observed_output = Hash_set.create (module String)
let failure_key ~source ~identity = source ^ ":" ^ identity

let observe_failure ~source ~format =
  let identity =
    Stdlib.Digest.string (Refusal_control_scan.normalize format) |> Stdlib.Digest.to_hex
  in
  Hash_set.add observed_failures (failure_key ~source ~identity)

let observe_output text =
  String.split_lines text |> List.map ~f:String.strip
  |> List.filter ~f:(Fn.non String.is_empty)
  |> List.iter ~f:(Hash_set.add observed_output)

let printf format =
  Printf.ksprintf
    (fun output ->
      observe_output output;
      Stdio.printf "%s%!" output)
    format

let claim_exercises passed_labels diagnostic =
  let rec take_match before = function
    | [] -> None
    | label :: after ->
        if Refusal_control_scan.format_matches ~format:diagnostic.Refusal_control_scan.format label
        then Some (List.rev_append before after)
        else take_match (label :: before) after
  in
  take_match [] passed_labels

let evidence_observed ~passed_labels evidence =
  Hash_set.mem observed_output evidence
  ||
  match String.chop_suffix evidence ~suffix:": true" with
  | Some label -> List.mem passed_labels label ~equal:String.equal
  | None -> List.mem passed_labels evidence ~equal:String.equal

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
        if List.Assoc.mem entries local ~equal:String.equal then local else normalized
  in
  let diagnostics = Refusal_control_scan.diagnostics (In_channel.read_all source_path) in
  let expected = markers source in
  let passed_labels = ref (Verdict.passed_labels ()) in
  printf "\nSynthetic controls: scanner refusal diagnostics exercised by this control golden:\n";
  diagnostics
  |> List.iter ~f:(fun diagnostic ->
      let marker = Refusal_control_scan.marker diagnostic in
      if List.mem expected marker ~equal:String.equal then
        match diagnostic.Refusal_control_scan.kind with
        | Refusal_control_scan.Claim -> (
            match claim_exercises !passed_labels diagnostic with
            | None -> ()
            | Some remaining ->
                passed_labels := remaining;
                printf "  %s\n" marker)
        | Refusal_control_scan.Fail ->
            let key = failure_key ~source ~identity:diagnostic.Refusal_control_scan.identity in
            let assigned_evidence = List.Assoc.find direct_evidence key ~equal:String.equal in
            if
              Hash_set.mem observed_failures key
              || Option.value_map assigned_evidence ~default:false
                   ~f:(evidence_observed ~passed_labels:(Verdict.passed_labels ()))
            then printf "  %s\n" marker)
