import matplotlib.pyplot as plt
import numpy as np
import threading

class TopDownPlot:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))

        self.path_x = []
        self.path_y = []

        self.robot_marker, = self.ax.plot([], [], 'bo')
        self.heading_line, = self.ax.plot([], [], 'b-')
        self.path_line, = self.ax.plot([], [], 'g-')

        self.table_patch = None
        self.target_marker, = self.ax.plot([], [], 'ro')

        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_aspect('equal')
        self.ax.grid(True)

        self.lock = threading.Lock()

    def update(self, pose, target=None, table=None, table_size=None, table_yaw=0.0, obstacles=[], waypoint=None):
        with self.lock:
            x, y, theta = pose

            # Store trail
            self.path_x.append(x)
            self.path_y.append(y)

            # Update robot marker
            self.robot_marker.set_data([x], [y])

            # Update heading arrow
            hx = x + 0.4*np.cos(theta)
            hy = y + 0.4*np.sin(theta)
            self.heading_line.set_data([x, hx], [y, hy])

            # Update path
            self.path_line.set_data(self.path_x, self.path_y)

            # Update target
            if target is not None:
                self.target_marker.set_data([target[0]], [target[1]])

            # Draw table rectangle
            if table is not None and table_size is not None:
                if self.table_patch is not None:
                    self.table_patch.remove()

                from matplotlib.patches import Rectangle
                from matplotlib.transforms import Affine2D

                hx = table_size[0] / 2
                hy = table_size[1] / 2

                rect = Rectangle((-hx, -hy), table_size[0], table_size[1],
                                 linewidth=1, edgecolor='brown', facecolor='none')

                t = Affine2D().rotate(table_yaw).translate(table[0], table[1])
                rect.set_transform(t + self.ax.transData)

                self.ax.add_patch(rect)
                self.table_patch = rect
            
            if waypoint:
                wx, wy = waypoint
                self.ax.plot(wx, wy, 'mx', markersize=10, label="Waypoint")
            
            for obs in obstacles:
                ox, oy = obs[:2]
                circle = plt.Circle((ox, oy), 0.2, color='black', fill=False)
                self.ax.add_patch(circle)
            


            self.fig.canvas.draw()
            self.fig.canvas.flush_events()