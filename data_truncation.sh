#do not use shuffle as will not be the same for others

original_accepted="Data/Lending_club/accepted_2007_to_2018Q4.csv"
original_rejected="Data/Lending_club/rejected_2007_to_2018Q4.csv"

target_accepted="data/Lending_club/sample_accepted_2007_to_2018Q4.csv"
target_rejected="data/Lending_club/sample_rejected_2007_to_2018Q4.csv"

#print("Sample accepted csv location: $target_accepted")
#print("Sample rejected csv location: $target_rejected")

# Extract and store the headers
head -n 1 $original_accepted > $target_accepted
head -n 1 $original_rejected > $target_rejected

# Extract and store the data
head -n 20001 $original_accepted | tail -n 20000 >> $target_accepted
head -n 20001 $original_rejected | tail -n 20000 >> $target_rejected



#shuf -n 501 Data/Lending_club/accepted_2007_to_2018Q4.csv > $target_accepted
#shuf -n 501 Data/Lending_club/rejected_2007_to_2018Q4.csv > $target_rejected