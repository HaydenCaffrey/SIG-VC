
mkdir -p ../bn_extractor/data/debug

python prepare_bnf_input.py $1 ../bn_extractor/data/debug || exit 1

cd ../bn_extractor
ls
bash extract_btlnfea.sh data/debug data/debug/ivector data/debug/bnf || exit 1

cd ../conversion
python decompress_ark.py ../bn_extractor/data/debug/bnf $2 || exit 1
