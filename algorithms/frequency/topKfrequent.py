"""
Problem:
Given a list of integers nums and an integer k, return the k most frequent elements in the list.
You cannot use collections.Counter — implement the logic yourself.
The order of the returned list does not matter.

Example 1
nums = [1,1,1,2,2,3]
k = 2
Output: [1, 2]

Example 2
nums = [4,4,1,2,2,3]
k = 1
Output: [4] or [2]  # both valid if tied, but 4 is most frequent
"""

def topKfrequent(nums: list, k: int) -> list:
    #we are going to use a hash map
    #edge cases
    if k == 0:
        return []
    if len(nums) == 0:
        return nums
    newList = {}
    #populates dict with each numbers frequency
    for number in nums:
        newList[number] = newList.get(number, 0) + 1
    
    final = []
    for _ in range(0, k):
        val = max(newList, key=lambda x: newList[x])
        final.append(val)
        newList.pop(val)
    return final


def main():
    newlist = [1,1,1,2,2,3]
    k = 2
    print(topKfrequent(newlist, k))

    newlist = [4,4,1,2,2,3]
    k = 1
    print(topKfrequent(newlist, k))
    
    newlist = [1]
    k = 1
    print(topKfrequent(newlist, k))

    newlist = [1,1,1,1,2,3]
    k = 3
    print(topKfrequent(newlist, k))

if __name__ == "__main__":
    main()

    
