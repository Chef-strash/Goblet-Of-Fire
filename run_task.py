import subprocess

# Train the model
print("Running file that trains the model")
subprocess.run(["python", "harry_q_learner.py"])

# Run the game 
print("Running file that renders the game environment")
subprocess.run(["python", "GOF_auto_play.py"])
