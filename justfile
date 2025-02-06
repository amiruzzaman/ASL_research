_default:
    @just --list --unsorted --justfile {{justfile()}}

dev:
    nix develop --command nix run .#_dev

