{"buggy_line": "    greater = quicksort([x for x in arr[1:] if x > pivot])\n", "correct_line": "    greater = quicksort([x for x in arr[1:] if x >= pivot])\n"}