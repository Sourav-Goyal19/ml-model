from manim import *
import random

class ExplainLinearRegression(Scene):
    def construct(self):
        # 1. Introduction & Scatter Plot
        axes = Axes(
            x_range=[0, 35, 5],
            y_range=[0, 30, 5],
            x_length=7,
            y_length=5,
            axis_config={"include_numbers": True}
        )
        self.play(Create(axes), run_time=1)
        title = Text("Linear Regression: Predicting Ice Cream Sales").to_edge(UP)
        self.play(Write(title), run_time=1)
        data_points = []
        x_values = [10, 15, 20, 25, 30]
        y_values = [12, 15, 20, 25, 28]
        for i in range(len(x_values)):
            dot = Dot(color=BLUE).move_to(axes.c2p(x_values[i], y_values[i]))
            data_points.append(dot)
        self.play(*[Create(dot) for dot in data_points], run_time=2)

        # 2. "Best-fit" Line Animation
        initial_line = Line(axes.c2p(0, 5), axes.c2p(35, 25), color=RED)
        optimal_line = Line(axes.c2p(0, 8), axes.c2p(35, 28), color=RED)
        self.play(Create(initial_line))
        self.play(Transform(initial_line, optimal_line), run_time=2)

        # 3. Distances from points to line
        # ... (Add lines/arrows representing distances) ...

        # 4. Cost Function
        cost_text = Text("Cost Function (MSE) Decreasing").to_edge(DOWN + RIGHT)
        cost_value = Text("100").next_to(cost_text, DOWN)
        self.play(Write(cost_text), Write(cost_value))
        # ... (Animate cost_value decreasing) ...


        # 5. Equation of a line
        equation = Text("y = mx + c").to_edge(UP+LEFT)
        self.play(Write(equation))


        # 6. Prediction
        # ... (Animate prediction process) ...

        # 7. Non-linear Relationship
        # ... (Animate non-linear curve) ...

        # 8. Summary & Call to Action
        summary = Text("Key Concepts: Correlation, Best-Fit Line, Cost Function, Prediction, Limitations").to_edge(DOWN)
        call_to_action = Text("Learn More & Practice!").to_edge(DOWN)
        self.play(Write(summary), run_time=2)
        self.play(Write(call_to_action), run_time=1)
        self.wait(2)