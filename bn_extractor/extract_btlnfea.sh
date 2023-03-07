. ./cmd.sh
. ./path.sh

nj=1
mfcc_nj=3
stage=0
ivector_nj=1

. parse_options.sh || exit 1;

data_dir=$1
ivec_dir=$2
tgt_dir=$3

if [[ ( $# -ne 3 ) ]]; then
   echo "usage: $0 <input-data-dir> <ivector-data-dir> <target-dir> "
   echo "e.g.:  $0 data/test_clean_hires exp/nnet3_cleaned/ivectors_test_clean_hires data/test_clean_prefinalChain_bnf"
   echo "main options (for others, see top of script file)"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

if [ $stage -le 0 ]; then

utils/fix_data_dir.sh $data_dir || exit 1;

steps/make_mfcc.sh --mfcc-config mfcc_hires.conf --nj $mfcc_nj \
  $data_dir make_mfcc mfcc || exit 1;

steps/compute_cmvn_stats.sh $data_dir make_mfcc mfcc || exit 1;

utils/fix_data_dir.sh $data_dir || exit 1;

fi


if [ $stage -le 1 ]; then

steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $ivector_nj \
             $data_dir ivector_extractor $ivec_dir


fi

if [ $stage -le 2 ]; then

./chain_get_bottleneck_features.sh --nj $nj --ivector-dir $ivec_dir \
                                   prefinal-l $data_dir $tgt_dir acoustic_mdl

fi

