from manim import *

class LinearRegression(Scene):
    def construct(self):
        # --- Data Points ---
        data_points = [
            (1, 2), (2, 3), (3, 5), (4, 4), (5, 6), (6, 7), (7, 9), (8,8)
        ]

        # --- Axes and Scatter Plot ---
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            x_length=7,
            y_length=5,
            axis_config={"include_numbers": True}
        )
        dots = VGroup(*[Dot(axes.c2p(x, y), color=BLUE) for x, y in data_points])
        
        # --- Best Fit Line ---
        def best_fit_line(points):
            x_coords = np.array([point[0] for point in points])
            y_coords = np.array([point[1] for point in points])
            A = np.vstack([x_coords, np.ones(len(x_coords))]).T
            m, c = np.linalg.lstsq(A, y_coords, rcond=None)[0]
            return m, c

        m, c = best_fit_line(data_points)
        line = Line(start=axes.c2p(0, c), end=axes.c2p(10, 10*m + c), color=RED)

        # --- Residuals ---
        residuals = VGroup()
        for x, y in data_points:
            point = axes.c2p(x,y)
            line_point = axes.c2p(x, m*x + c)
            residual = Line(point, line_point, color=YELLOW)
            residuals.add(residual)

        # --- Labels ---
        slope_label = MathTex("Slope (m) = ", str(round(m,2))).next_to(line, UP, buff=0.2)
        intercept_label = MathTex("Intercept (c) = ", str(round(c,2))).next_to(line, DOWN, buff=0.2)


        # --- Animation Sequences ---
        self.play(Create(axes))
        self.wait(0.5)
        for dot in dots:
            self.play(Create(dot), run_time=0.2)
        self.wait(0.5)
        self.play(Create(line), run_time=1)
        self.wait(0.5)
        self.play(Create(residuals), run_time=1)
        self.wait(0.5)
        self.play(Write(slope_label))
        self.play(Write(intercept_label))
        self.wait(2)

        # --- Title ---
        title = Tex("Linear Regression").scale(1.5).to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        

        self.wait(2)