nix
with import (builtins.fetchGit {
  url = "https://github.com/NixOS/nixpkgs";
  ref = "nixos-unstable";
}) { config.allowUnfree = true; };

mkShell {
  venvDir = "./_venv";

  buildInputs = [
    pkgs.python312
  ] ++ (with pkgs.python312Packages; [
    venvShellHook
    wheel
    pip
  ]) ++ (import ./system-dependencies.nix { inherit pkgs; });

  postShellHook = ''
    unset SOURCE_DATE_EPOCH
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc pkgs.zlib ]}:$LD_LIBRARY_PATH"
    export PYTHONPATH=$venvDir/${pkgs.python312.sitePackages}:$PYTHONPATH
    pip install -r requirements.txt
  '';
}