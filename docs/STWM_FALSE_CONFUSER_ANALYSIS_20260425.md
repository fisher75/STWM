# STWM False-Confuser Analysis 20260425

## Overall

| scope | count | teacher top1 | belief top1 | delta | teacher false-confuser | belief false-confuser |
|---|---:|---:|---:|---:|---:|---:|
| all_paired_trace_belief_rows | 3480 | 0.4845 | 0.5006 | 0.0161 | 0.5155 | 0.4994 |
| id_densified_200 | 894 | 0.4228 | 0.4631 | 0.0403 | 0.5772 | 0.5369 |
| true_ood_heldout | 2586 | 0.5058 | 0.5135 | 0.0077 | 0.4942 | 0.4865 |

## Groups

| group | count | teacher top1 | belief top1 | delta | teacher false-confuser | belief false-confuser | representative ids |
|---|---:|---:|---:|---:|---:|---:|---|
| teacher_high_conf_wrong | 618 | 0.0000 | 0.4563 | 0.4563 | 1.0000 | 0.5437 | burst::ArgoVerse::10b8dee6-778f-33e4-a946-d842d2d9c3d7::3, burst::BDD::b27688b6-1af27060::1, burst::BDD::b2eed8fa-2694b15d::2, burst::BDD::b2f68bad-aa8f35d1::4, burst::BDD::b306fb3f-f02e46cc::2 |
| belief_corrects_teacher | 523 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | burst::ArgoVerse::10b8dee6-778f-33e4-a946-d842d2d9c3d7::3, burst::ArgoVerse::rear_right_7dd530ed-80d9-30b7-80a6-57e7d334f302::1, burst::BDD::b27688b6-1af27060::1, burst::BDD::b2eed8fa-2694b15d::2, burst::BDD::b3e72283-d9fc39b0::5 |
| teacher_correct_belief_wrong | 467 | 1.0000 | 0.0000 | -1.0000 | 0.0000 | 1.0000 | burst::BDD::b27688b6-1af27060::2, burst::BDD::b2778280-4179c4af::3, burst::BDD::b37c86c4-53f2f54c::4, burst::Charades::5AE54::3, burst::Charades::VE6GK::4 |
| both_correct | 1219 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | burst::ArgoVerse::rear_right_00c561b9-2057-358d-82c6-5b06d76cebcf::1, burst::ArgoVerse::rear_right_9da4ca63-f524-3b38-8c8b-624f17518574::2, burst::BDD::b27688b6-1af27060::2, burst::BDD::b2778280-4179c4af::3, burst::BDD::b322412a-b47af37f::1 |
| both_wrong | 1271 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | burst::ArgoVerse::10b8dee6-778f-33e4-a946-d842d2d9c3d7::3, burst::ArgoVerse::45753856-4575-4575-4575-345754906624::3, burst::ArgoVerse::rear_right_02cf0ce1-699a-373b-86c0-eb6fd5f4697a::2, burst::ArgoVerse::side_left_043aeba7-14e5-3cde-8a5c-639389b6d3a6::2, burst::BDD::b2ed13f9-01b4dd4f::1 |
| continuity_heavy | 1440 | 0.5208 | 0.5188 | -0.0021 | 0.4792 | 0.4813 | burst::ArgoVerse::10b8dee6-778f-33e4-a946-d842d2d9c3d7::3, burst::ArgoVerse::rear_right_02cf0ce1-699a-373b-86c0-eb6fd5f4697a::2, burst::ArgoVerse::side_left_043aeba7-14e5-3cde-8a5c-639389b6d3a6::2, burst::BDD::b27688b6-1af27060::2, burst::BDD::b2ed13f9-01b4dd4f::1 |
| ambiguity_heavy | 2562 | 0.4239 | 0.4664 | 0.0425 | 0.5761 | 0.5336 | burst::ArgoVerse::10b8dee6-778f-33e4-a946-d842d2d9c3d7::3, burst::ArgoVerse::45753856-4575-4575-4575-345754906624::3, burst::ArgoVerse::rear_right_00c561b9-2057-358d-82c6-5b06d76cebcf::1, burst::ArgoVerse::rear_right_02cf0ce1-699a-373b-86c0-eb6fd5f4697a::2, burst::ArgoVerse::rear_right_7dd530ed-80d9-30b7-80a6-57e7d334f302::1 |
| OOD_hard | 2586 | 0.5058 | 0.5135 | 0.0077 | 0.4942 | 0.4865 | burst::ArgoVerse::043aeba7-14e5-3cde-8a5c-639389b6d3a6::3, burst::ArgoVerse::043aeba7-14e5-3cde-8a5c-639389b6d3a6::5, burst::ArgoVerse::10b8dee6-778f-33e4-a946-d842d2d9c3d7::1, burst::ArgoVerse::10b8dee6-778f-33e4-a946-d842d2d9c3d7::2, burst::ArgoVerse::10b8dee6-778f-33e4-a946-d842d2d9c3d7::3 |

false_confuser_reduced = `True`
