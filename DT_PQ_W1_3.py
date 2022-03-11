def hitung_binary(data, value):
    low = 0
    high = len(data)-1
    data.sort()
    a = 0
    while low <= high and not False:
        tengah = (low+high)//2
        if (value < data[tengah]):
            high = tengah - 1
            print ("Cek Kiri")
        
        elif (value == data[tengah]):
            return tengah

        elif (value > data[tengah]):
            low = tengah + 1
            print("Cek Kanan")

    return -1

print(hitung_binary([10, 2, 9, 6, 7, 1, 5, 3, 4, 8],1))