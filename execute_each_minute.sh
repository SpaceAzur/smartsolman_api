while true
do 
    timy=$(date +%S)
    if [ "$timy" -eq 45 ]
    then 
        date +%M:%S
        sleep 1s
    fi 
done
