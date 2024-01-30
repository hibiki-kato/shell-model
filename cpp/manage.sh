proc_num=10
for ((i=1; i<=proc_num; i++))
do
    pjsub run.sh $i $proc_nu
done
