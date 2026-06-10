with import (builtins.fetchGit {
  url = "https://github.com/NixOS/nixpkgs";
  # Pick a recent commit that includes Python 3.12
  ref = "nixos-unstable";
  # rev = "5df4d78a92474e6a45d8e6a9a77b6ca1f427c5c3"; # 2024-09-01
}) { config.allowUnfree = true; };

mkShell {
  venvDir = "./_venv";

  buildInputs = [
    pkgs.python312
  ] ++ (with pkgs.python312Packages; [
    venvShellHook
    wheel
  ]) ++ (import ./system-dependencies.nix { inherit pkgs; });

  postShellHook = ''
    unset SOURCE_DATE_EPOCH
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc pkgs.zlib ]}:$LD_LIBRARY_PATH"
    export PYTHONPATH=$venvDir/${pkgs.python312.sitePackages}:$PYTHONPATH
  '';

  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    pip install -r requirements.txt
  '';
}
