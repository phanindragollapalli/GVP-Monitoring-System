import serial
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SerialGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GUI")
        self.root.geometry("600x500")  # Set window size

        # Serial communication variables
        self.ser = None
        self.running = False
        self.queue = queue.Queue()
        self.temperatures = []  # For plotting and stats
        self.humidities = []    # For plotting and stats

        # GUI layout
        # Frame for input fields
        self.input_frame = ttk.Frame(root)
        self.input_frame.pack(pady=10)

        # COM port input
        ttk.Label(self.input_frame, text="COM Port:").grid(row=0, column=0, padx=5, pady=5)
        self.com_entry = ttk.Entry(self.input_frame, width=15)
        self.com_entry.grid(row=0, column=1, padx=5, pady=5)
        self.com_entry.insert(0, "COM3")  # Default

        # Baud rate input
        ttk.Label(self.input_frame, text="Baud Rate:").grid(row=1, column=0, padx=5, pady=5)
        self.baud_combo = ttk.Combobox(self.input_frame, values=["9600", "19200", "38400", "57600", "115200"], width=12)
        self.baud_combo.grid(row=1, column=1, padx=5, pady=5)
        self.baud_combo.set("9600")  # Default

        # Connect/Disconnect button
        self.connect_btn = ttk.Button(self.input_frame, text="Connect", command=self.toggle_connection)
        self.connect_btn.grid(row=2, column=0, columnspan=2, pady=10)

        # Text box for data display
        self.data_text = tk.Text(root, height=10, width=50, state='normal')
        self.data_text.pack(pady=10)

        # Plotting area (optional for Step 4)
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(pady=10)

        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def toggle_connection(self):
        if not self.running:
            # Try to connect
            com_port = self.com_entry.get()
            try:
                baud_rate = int(self.baud_combo.get())
            except ValueError:
                messagebox.showerror("Error", "Invalid baud rate!")
                return

            try:
                self.ser = serial.Serial(com_port, baud_rate, timeout=1)
                self.running = True
                self.connect_btn.config(text="Disconnect")
                self.com_entry.config(state='disabled')
                self.baud_combo.config(state='disabled')
                self.data_text.delete(1.0, tk.END)
                self.data_text.insert(tk.END, "Connected. Streaming data...\n")
                threading.Thread(target=self.read_serial, daemon=True).start()
                self.root.after(100, self.update_gui)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to connect: {e}")
        else:
            # Disconnect
            self.running = False
            self.connect_btn.config(text="Connect")
            self.com_entry.config(state='normal')
            self.baud_combo.config(state='normal')
            self.data_text.insert(tk.END, "Disconnected.\n")
            if self.ser:
                self.ser.close()
                self.ser = None

    def read_serial(self):
        while self.running:
            if self.ser and self.ser.in_waiting > 0:
                try:
                    line = self.ser.readline().decode('utf-8').rstrip()
                    self.queue.put(line)
                except Exception as e:
                    self.queue.put(f"Error reading data: {e}")

    def update_gui(self):
        try:
            while not self.queue.empty():
                line = self.queue.get_nowait()
                self.data_text.insert(tk.END, line + "\n")
                self.data_text.see(tk.END)  # Auto-scroll

                # Parse data for plotting and stats (for DHT22)
                if "Temperature" in line:
                    try:
                        temp = float(line.split("Temperature: ")[1].split(" C")[0])
                        hum = float(line.split("Humidity: ")[1].split(" %")[0])
                        self.temperatures.append(temp)
                        self.humidities.append(hum)
                        if len(self.temperatures) > 50:  # Limit to 50 points
                            self.temperatures.pop(0)
                            self.humidities.pop(0)

                        # Update stats
                        if self.temperatures:
                            avg_temp = sum(self.temperatures) / len(self.temperatures)
                            min_temp = min(self.temperatures)
                            max_temp = max(self.temperatures)
                            self.data_text.insert(tk.END, f"Avg Temp: {avg_temp:.2f} C, Min: {min_temp:.2f} C, Max: {max_temp:.2f} C\n")

                        # Update plot
                        self.ax.clear()
                        self.ax.plot(self.temperatures, label="Temperature (°C)")
                        self.ax.plot(self.humidities, label="Humidity (%)")
                        self.ax.legend()
                        self.ax.set_title("Sensor Data Over Time")
                        self.canvas.draw()
                    except ValueError:
                        pass  # Skip malformed data
        except queue.Empty:
            pass
        if self.running:
            self.root.after(100, self.update_gui)

    def on_closing(self):
        self.running = False
        if self.ser:
            self.ser.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SerialGUI(root)
    root.mainloop()