input_dir=$1
output=output
filelist=`ls $input_dir` 
for file in $filelist
do
  echo $input_dir$file
  python -m apps.prt_util -i $input_dir$file
  python -m apps.render_data -i $input_dir$file -o $output -e
done
