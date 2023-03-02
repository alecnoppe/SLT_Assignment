import assignments.A1a
import assignments.A1a_sklearn
import assignments.A1b
import assignments.A1c
import sys

if __name__ == "__main__":
    args = sys.argv
    sub_question = int(args[1])
    if sub_question == 1:
        assignments.A1a.main()
    elif sub_question == 2:
        assignments.A1b.main()
    elif sub_question == 3:
        assignments.A1c.main()
    elif sub_question == -1: #Sklearn implementation
        assignments.A1a_sklearn.main()
else:
    print("Run in CLI, give 1 argument for which script")