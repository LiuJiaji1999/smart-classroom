null = [{'seatListArea': 
    [{'x': 234, 'y': 1054}, 
    {'x': 962, 'y': 1434},
    {'x': 1366, 'y': 1202}, 
    {'x': 580, 'y': 844}], 
    'seatNum': 4, 
    'userIdList': 
    ['null', '1cfd0bd9231dc85bb0986b36b595118d', 'null', '0fdf9b15f6c8f4d3283fde17e66ab534']}, 
{'seatListArea': 
    [{'x': 1344, 'y': 444}, {'x': 1942, 'y': 608}, {'x': 2028, 'y': 500}, {'x': 1504, 'y': 362}], 
    'seatNum': 4, 
    'userIdList': ['null', 'ab27ea520d7597aa7e7bd93b612d580d', 'null', '5ed04da53a49e62a3dc03ca51cf74522']}, 
{'seatListArea': [{'x': 874, 'y': 680}, {'x': 1628, 'y': 946}, {'x': 1824, 'y': 728}, {'x': 1154, 'y': 530}], 
    'seatNum': 4, 
    'userIdList': ['null', '0db9281430aae456b5691ba907293a2b', 'null', '65dc4b5963f3311c3b4d1f05342b90ce']}, 
{'seatListArea': [{'x': 1622, 'y': 308}, {'x': 2102, 'y': 434}, {'x': 2152, 'y': 372}, {'x': 1736, 'y': 270}], 
    'seatNum': 4, 
    'userIdList': ['null', 'null', 'null', 'null']}, 
{'seatListArea': [{'x': 578, 'y': 850}, {'x': 1358, 'y': 1196}, {'x': 1626, 'y': 946}, {'x': 886, 'y': 684}], 
    'seatNum': 4, 'userIdList': ['null', '2db30e0b62d964ccd88baac198bb4c4d', 'null', '1be6bc4abc46bd3f4b3bd70138848f6f']}, 
{'seatListArea': [{'x': 1502, 'y': 362}, {'x': 2028, 'y': 504}, {'x': 2114, 'y': 426}, {'x': 1654, 'y': 308}], 
    'seatNum': 4, 
    'userIdList': ['null', 'null', 'null', 'null']}, 
{'seatListArea': [{'x': 1156, 'y': 532}, {'x': 1816, 'y': 736}, {'x': 1956, 'y': 600}, {'x': 1354, 'y': 440}], 
    'seatNum': 4, 
    'userIdList': ['null', 'b54a74bc6434f475e61dadb97e41ccdb', 'null', '3a3ce50d5ecc992803db0e68b17bc63e']}]
# print(type(null))
# print(len(null))
users = []
for i in range(len(null)):
    # print(null[i])
    # print(type(null[i]))
    # print(null[i]['userIdList'])
    for j in range(len(null[i]['userIdList'])):
        if null[i]['userIdList'][j] != 'null':
            users.append(null[i]['userIdList'][j])
print(users)
print(len(users))

