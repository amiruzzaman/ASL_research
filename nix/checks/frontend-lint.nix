{pkgs, ...}:
pkgs.frontend.overrideAttrs (old: {
  name = "frontend-lint";
  installPhase = "mkdir $out;";
  npmBuildScript = "lint";
})
