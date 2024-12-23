import tkinter as tk
from tkinter import messagebox
import pandas as pd
import bcrypt
import subprocess


def check_user(username, password, email):
    try:
        users_df = pd.read_csv("users.csv")
    except FileNotFoundError:
        users_df = pd.DataFrame(columns=["Username", "Password", "Email"])

    if "Username" not in users_df.columns:
        users_df["Username"] = []
    if "Password" not in users_df.columns:
        users_df["Password"] = []
    if "Email" not in users_df.columns:
        users_df["Email"] = []

    if username in users_df["Username"].values:
        return False
    else:
        hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        new_user = pd.DataFrame(
            {
                "Username": [username],
                "Password": [hashed_password.decode("utf-8")],
                "Email": [email],
            }
        )
        users_df = pd.concat([users_df, new_user], ignore_index=True)
        users_df.to_csv("users.csv", index=False)
        return True


def login():
    username = entry_username.get()
    password = entry_password.get()

    if username == "" or password == "":
        footer_label.config(text="Please fill in both fields", fg="red")
    else:
        try:
            users_df = pd.read_csv("users.csv")

            if username in users_df["Username"].values:
                stored_hashed_password = users_df[users_df["Username"] == username][
                    "Password"
                ].values[0]
                if bcrypt.checkpw(
                    password.encode("utf-8"), stored_hashed_password.encode("utf-8")
                ):
                    footer_label.config(
                        text=f"Login Success: Welcome, {username}!", fg="green"
                    )
                    go_to_predictions_button.pack(pady=10)
                else:
                    footer_label.config(text="Incorrect password", fg="red")
            else:
                footer_label.config(text="User not found", fg="red")
        except FileNotFoundError:
            footer_label.config(
                text="No user data found. Please register first.", fg="red"
            )


def create_account():
    username = entry_username.get()
    password = entry_password.get()
    email = entry_email.get()

    if username == "" or password == "" or email == "":
        footer_label.config(text="Please fill in all fields", fg="red")
    else:
        if check_user(username, password, email):
            footer_label.config(
                text=f"Account Created: User {username} created successfully!",
                fg="green",
            )
        else:
            footer_label.config(text="Username already exists", fg="red")


def go_to_predictions():
    footer_label.config(text="Going to predictions...", fg="blue")
    try:
        subprocess.run(["python3", "main.py"], check=True)
    except subprocess.CalledProcessError as e:
        footer_label.config(text=f"Error: {e}", fg="red")
    except FileNotFoundError:
        footer_label.config(text="Error: main.py file not found.", fg="red")


root = tk.Tk()
root.title("User Login and Registration")

header_frame = tk.Frame(root, bg="#3b5998", height=100)
header_frame.grid(row=0, column=0, columnspan=2, sticky="ew")

label_title = tk.Label(
    header_frame,
    text="User Login & Registration",
    font=("Helvetica", 24),
    fg="white",
    bg="#3b5998",
)
label_title.pack(pady=20)

body_frame = tk.Frame(root)
body_frame.grid(row=1, column=0, columnspan=2, pady=20)

label_username = tk.Label(body_frame, text="Username")
label_username.grid(row=0, column=0)

entry_username = tk.Entry(body_frame)
entry_username.grid(row=0, column=1)

label_password = tk.Label(body_frame, text="Password")
label_password.grid(row=1, column=0)

entry_password = tk.Entry(body_frame, show="*")
entry_password.grid(row=1, column=1)

label_email = tk.Label(body_frame, text="Email")
label_email.grid(row=2, column=0)

entry_email = tk.Entry(body_frame)
entry_email.grid(row=2, column=1)

login_button = tk.Button(
    body_frame,
    text="Login",
    command=login,
    bg="#4CAF50",
    fg="#3b5998",
    font=("Helvetica", 14),
)
login_button.grid(row=3, column=0, pady=10)

create_button = tk.Button(
    body_frame,
    text="Create Account",
    command=create_account,
    bg="#3b5998",
    fg="#3b5998",
    font=("Helvetica", 14),
)
create_button.grid(row=3, column=1, pady=10)

footer_frame = tk.Frame(root, bg="#f1f1f1", height=100)
footer_frame.grid(row=2, column=0, columnspan=2, sticky="ew")

footer_label = tk.Label(footer_frame, text="", font=("Helvetica", 12), bg="#f1f1f1")
footer_label.pack(pady=20)

go_to_predictions_button = tk.Button(
    footer_frame,
    text="Go to Predictions",
    command=go_to_predictions,
    bg="#4CAF50",
    fg="#3b5998",
    font=("Helvetica", 14),
)
go_to_predictions_button.pack_forget()

root.mainloop()