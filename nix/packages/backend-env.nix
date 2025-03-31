{
  pkgs,
  outputs,
}:
(outputs.lib.pythonSetForPkgs pkgs).mkVirtualEnv "asl-research-env" outputs.lib.workspace.deps.default
