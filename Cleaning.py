import csv

input = open('MovieI.csv', 'rb')
output = open('MovieO.csv', 'wb')
writer = csv.writer(output)
for row in csv.reader(input):
    for i in range(len(row)):
        if(row[0]==''):
            break
        elif(row[1]==''):
            break
        elif(row[2]==''):
            break
        elif(row[3]==''):
            break
        elif(row[4]==''):
            break
    else :writer.writerow(row)

input.close()
output.close()