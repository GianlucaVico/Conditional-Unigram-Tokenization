#!/bin/bash
# $1 output dir
# $2 reference translation
# $3 translatation output
# Called as ./validate.sh args translation-output
# ? | detokenize | BLEU score
# echo "$0\n$1\n$2\n$3\n" > validation_debug.txt
cat $2 | sed 's/ //g ; s/▁/ /g' > "$1/temp.detok"

cat $3 | sed 's/ //g ; s/▁/ /g' | sacrebleu "$1/temp.detok" -m bleu -b -w 4 --force >> "$1/validation.bleu"
cat $3 | sed 's/ //g ; s/▁/ /g' | sacrebleu "$1/temp.detok" -m ter -b -w 4 --force >> "$1/validation.ter"
cat $3 | sed 's/ //g ; s/▁/ /g' | sacrebleu "$1/temp.detok" -m chrf -b -w 4 -cw 2 --force >> "$1/validation.chrf"

cat $3 | sed 's/ //g ; s/▁/ /g' | sacrebleu "$1/temp.detok" -m bleu -b -w 4 --force # Suppress message
rm "$1/temp.detok"
