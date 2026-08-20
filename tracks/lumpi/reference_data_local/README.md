Dummy reference data for `local_eval.py`. Contains 5 token frames so the
pipeline runs end-to-end; AP will be 0 because detection_gt.json is empty,
and realism scores will be on a 5-frame reference (not 50). To reproduce
server-faithful realism scores, copy 50 real frames you've held out
yourself (and that are NOT in forbidden_frames.txt) into
realism_reference/, then update n_realism_frames in config.json to 50.
