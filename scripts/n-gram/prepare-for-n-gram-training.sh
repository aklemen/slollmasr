apt install libeigen3-dev
cd "$HOME"/local || exit
git clone https://github.com/kpu/kenlm.git
cd kenlm && mkdir build && cd build && cmake .. && make -j

pip3 install git+https://github.com/kpu/kenlm.git
pip3 install git+https://github.com/flashlight/text.git