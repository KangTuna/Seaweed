import sys

input = sys.stdin.readline

n = int(input().rstrip())

dp = [0,0,1]

for i in range(3,n+1):
    new = []
    new.append(dp[i-1] + 1)
    if i%3 == 0:
        new.append(dp[i//3] + 1)
    if i%2 == 0:
        new.append(dp[i//2] + 1)
    dp.append(min(new))
    
print(dp[n])
