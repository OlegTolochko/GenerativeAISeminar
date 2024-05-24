from manim import *
from manim import config

# Set the background color
config.background_color = "#0D1219"
#Config for white background:
#config.background_color = WHITE
#Tex.set_default(color=BLACK)
#MathTex.set_default(color=BLACK)
#Text.set_default(color=BLACK)

def Text2(t):
    return Text(t,font="Poppins")

def addWatermark():
    watermark = ImageMobject("resources/images/University_LMU_background.png")

    # Scale and position the watermark
    watermark.to_edge(DOWN).to_edge(RIGHT)
    watermark.shift(DOWN * 0.5)
    watermark.shift(RIGHT * 0.5)
    return watermark

class VQGANStructure(Scene):
    def construct(self):
        # Colors
        encoder_input_color = GREEN
        decoder_output_color = YELLOW
        encoder_color = WHITE
        z_color = BLUE
        vq_color = PURPLE
        codebook_color = BLUE_E
        decoder_color = WHITE

        num_rows = 8
        table_data = [["a"] for _ in range(num_rows)]

        # Add the watermark to the background
        #self.add(addWatermark())

        # Create the table
        encoder_table = Table(
            table_data,
            include_outer_lines=True,
            line_config={"color": encoder_input_color},
            v_buff = 1.3
        ).shift(LEFT * 4)
        encoder_table.get_columns().set_opacity(0)
        encoder_table.scale(0.25)

        # Fade in the table
        self.play(FadeIn(encoder_table))

        self.wait(2)

        # Encoder
        encoder = Polygon(
            [-3.5, -1.5, 0], [-2, -0.5, 0], [-2, 0.5, 0], [-3.5, 1.5, 0],
            color=encoder_color, fill_opacity=0
        )
        encoder_text = Text("Encoder").scale(0.5).move_to(encoder.get_center())
        self.play(FadeIn(encoder), Write(encoder_text))

        # Latent space z
        z = Rectangle(width=0.5, height=1, color=z_color, fill_opacity=0).shift(LEFT * 1.25)
        z_text = Text("z").scale(0.5).move_to(z.get_center())
        self.play(FadeIn(z), Write(z_text))

        # Vector Quantization
        vq = Rectangle(width=2, height=1, color=vq_color, fill_opacity=0).shift(RIGHT * 0.5)
        vq_text = Paragraph('Vector','Quantization', alignment='center').scale(0.45).move_to(vq.get_center())
        self.play(FadeIn(vq), Write(vq_text))

        # Quantized latent space z'
        z_quant = Rectangle(width=0.5, height=1, color=z_color, fill_opacity=0).shift(RIGHT * 2.25)
        z_quant_text = Text("z'").scale(0.5).move_to(z_quant.get_center())
        self.play(FadeIn(z_quant), Write(z_quant_text))

        # Decoder
        decoder = Polygon(
            [3, -0.5, 0], [4.5, -1.5, 0], [4.5, 1.5, 0], [3, 0.5, 0],
            color=decoder_color, fill_opacity=0
        )
        decoder_text = Text("Decoder").scale(0.5).move_to(decoder.get_center())
        self.play(FadeIn(decoder), Write(decoder_text))

        table_data = [["a"] for _ in range(num_rows)]

        # Create the table
        decoder_table = Table(
            table_data,
            include_outer_lines=True,
            line_config={"color": decoder_output_color},
            v_buff=1.3
        ).shift(RIGHT * 5)
        decoder_table.get_columns().set_opacity(0)
        decoder_table.scale(0.25)

        self.play(FadeIn(decoder_table))

        self.wait(2)

        # Codebook
        codebook = Rectangle(width=2.5, height=1, color=codebook_color, fill_opacity=0).shift(UP * 2, RIGHT * 0.5)
        codebook_text = Text("Codebook").scale(0.5).move_to(codebook.get_center())
        self.play(FadeIn(codebook), Write(codebook_text))

        self.wait(2)

        # Load the image
        image = ImageMobject("resources/images/iris.jpg")
        image.scale(0.4).shift(LEFT * 5.5)  # Adjust the scaling and position as needed

        # Display the image
        self.play(FadeIn(image))

        # Move the image to the middle of the table and shrink it
        self.play(image.animate.move_to(encoder_table.get_center()).scale(0))
        self.remove(image)

        values = ["3.7", "2.2", "1.5", "0.5", "2.8", "4.6", "1.2", "2.2"]

        # Create text objects for the values at the center of the table
        value_texts = [Text(value).scale(0.25).move_to(encoder_table.get_center()) for value in values]

        vector_values = [[3.3], [7.5], [5.3], [1.3]]

        # Create the vector using the Matrix class
        z_vector = Matrix(vector_values, v_buff=0.5, bracket_h_buff=0.2)
        z_vector.scale(0.5)
        z_vector.move_to(z.get_center())

        # Animate the values moving to each cell simultaneously
        animations = []
        for i, value_text in enumerate(value_texts):
            cell = encoder_table.get_cell((i + 1, 1))
            animations.append(value_text.animate.move_to(cell.get_center()))
        encoder_z_arrow = Arrow(encoder.get_right(), z_vector.get_left(), buff=0.1)
        self.play(*animations, GrowArrow(encoder_z_arrow))

        # Ensure the value_texts are properly added to the table cells
        for i, value_text in enumerate(value_texts):
            cell = encoder_table.get_cell((i + 1, 1))
            value_text.move_to(cell.get_center())  # Ensure the text is correctly positioned in the cell
            self.add(value_text)
            encoder_table.add(value_text)  # Add the text to the table

        self.wait(2)

        # Create copies of the values to animate them moving somewhere else
        moving_values = [value_text.copy() for value_text in value_texts]
        new_location = z.get_center()
        move_animations = [value.animate.move_to(new_location) for value in moving_values]

        self.play(*move_animations, z_text.animate.next_to(z, DOWN))
        for value in moving_values:
            self.remove(value)

        # Smoothly replace z with z_vector
        self.play(Transform(z, z_vector))

        self.wait(2)

        zoom_group = VGroup(z_vector, encoder_table, z, encoder, z_text, encoder_text, vq, vq_text, z_quant_text, z_quant, codebook_text, codebook, encoder_z_arrow)
        zoom_center = zoom_group.get_center()

        fade_out_group = VGroup(decoder,decoder_text,decoder_table)

        # Zoom into the left part (scale and move)
        self.play(
            zoom_group.animate.scale(1.5),
            FadeOut(fade_out_group)
        )

        self.wait(2)

        self.play(
            codebook_text.animate.next_to(codebook, UP)
        )

        self.wait(2)

        # Define the 4 vectors with arbitrary values
        vector1 = [[1.1], [2.2], [3.3], [4.4]]
        vector2 = [[2.2], [3.3], [4.4], [5.5]]
        vector3 = [[3.3], [4.4], [5.5], [6.6]]
        vector4 = [[4.4], [5.5], [6.6], [7.7]]

        # Create the vectors using the Matrix class
        matrix1 = Matrix(vector1, v_buff=0.5, bracket_h_buff=0.2).scale(0.5)
        matrix2 = Matrix(vector2, v_buff=0.5, bracket_h_buff=0.2).scale(0.5)
        matrix3 = Matrix(vector3, v_buff=0.5, bracket_h_buff=0.2).scale(0.5)
        matrix4 = Matrix(vector4, v_buff=0.5, bracket_h_buff=0.2).scale(0.5)

        # Arrange the vectors next to each other
        vectors = VGroup(matrix1, matrix2, matrix3, matrix4).arrange(RIGHT, buff=0.2)

        vectors.move_to(codebook.get_center())

        # Add the vectors to the scene
        self.play(FadeIn(vectors))

        self.wait(2)

        z_vq_arrow = Arrow(z.get_right(), vq.get_left(), buff=0.1)
        self.play(GrowArrow(z_vq_arrow))

        self.wait(1)

        vq_matrix1__arrow = Arrow(vq.get_top(), matrix1.get_bottom(), buff=0.1)
        self.play(GrowArrow(vq_matrix1__arrow))

        self.wait(2)

        self.play(z_quant_text.animate.next_to(z_quant, DOWN))
        self.wait(2)

        matrix1_copy = matrix1.copy()
        zoom_group.add(matrix1_copy)

        self.play(matrix1_copy.animate.move_to(z_quant.get_center()).scale(1.5), FadeOut(z_quant))
        vq_z_quant_arrow = Arrow(vq.get_right(), matrix1_copy.get_left(), buff=0.1)
        self.play(GrowArrow(vq_z_quant_arrow))
        self.wait(2)

        self.play(FadeOut(vq_matrix1__arrow))
        zoom_group.add(vectors, vq_z_quant_arrow, z_vq_arrow)

        self.remove(z_quant)
        zoom_group.remove(z_quant)
        self.wait(2)
        self.play(zoom_group.animate.scale(1/1.5).move_to(zoom_center + (codebook_text.get_top() - codebook_text.get_bottom())/2), FadeIn(fade_out_group))

        self.wait(2)
        matrix1_decoder_arrow = Arrow(matrix1_copy.get_right(), decoder.get_left(), buff=0.1)
        self.play(GrowArrow(matrix1_decoder_arrow))
        self.wait(2)

        #matrix1_copy_copy = matrix1_copy.copy()
        #self.play(matrix1_copy_copy.animate.move_to(decoder_table.get_center()).scale(0))
        #self.remove(matrix1_copy_copy)

        values = ["3.9", "2.5", "1.3", "0.4", "3.4", "3.9", "1.2", "2.0"]

        # Create text objects for the values at the center of the table
        value_texts = [Text(value).scale(0.25).move_to(matrix1_copy.get_center()) for value in values]

        # Animate the values moving to each cell simultaneously
        animations = []
        for i, value_text in enumerate(value_texts):
            cell = decoder_table.get_cell((i + 1, 1))
            animations.append(value_text.animate.move_to(cell.get_center()))
        self.play(*animations)

        # Ensure the value_texts are properly added to the table cells
        for i, value_text in enumerate(value_texts):
            cell = decoder_table.get_cell((i + 1, 1))
            value_text.move_to(cell.get_center())  # Ensure the text is correctly positioned in the cell
            self.add(value_text)
            decoder_table.add(value_text)  # Add the text to the table

        self.wait(2)


import random

class PixelGrid(VGroup):
    def __init__(self, rows=4, cols=4, cell_size=1, **kwargs):
        super().__init__(**kwargs)

        # Create a grid of squares with random colors
        for i in range(rows):
            for j in range(cols):
                # Create a square
                square = Square(side_length=cell_size)
                # Set the position of the square
                square.move_to(np.array([j * cell_size, -i * cell_size, 0]))
                # Generate a random color
                random_color = self.random_bright_color()
                # Set the color of the square
                square.set_fill(random_color, opacity=1.0)
                square.set_stroke(width=0)  # Remove the stroke
                # Add the square to the grid
                self.add(square)

        # Center the grid
        self.move_to(ORIGIN)

    # Utility function to generate a random bright color
    def random_bright_color(self):
        return "#" + ''.join([random.choice('89ABCDEF') for _ in range(6)])


class StageCTraining(Scene):
    def construct(self):
        efficient_net_color = BLUE
        efficient_net_table_color = WHITE


        num_rows = 6
        table_data = [["a"] for _ in range(num_rows)]

        # Add the watermark to the background
        #self.add(addWatermark())

        # Create the table
        efficient_net_table = Table(
            table_data,
            include_outer_lines=True,
            line_config={"color": efficient_net_table_color},
            v_buff = 1.3
        ).shift(LEFT * 4)
        efficient_net_table.get_columns().set_opacity(0)
        efficient_net_table.scale(0.25)

        # Fade in the table
        self.play(FadeIn(efficient_net_table))

        self.wait(2)

        # Encoder
        efficient_net = Polygon(
            [-3.5, -1.25, 0], [-2, -0.5, 0], [-2, 0.5, 0], [-3.5, 1.25, 0],
            color=efficient_net_color, fill_opacity=0
        )
        efficient_net_text = Text("EfficientNet").scale(0.4).move_to(efficient_net.get_center())
        self.play(FadeIn(efficient_net), Write(efficient_net_text))

        efficient_net_group = VGroup(efficient_net_text, efficient_net, efficient_net_table)

        self.wait(2)
        self.play(efficient_net_group.animate.shift(RIGHT*2))

        self.wait(2)

        database_image = SVGMobject("resources/images/database-solid.svg")
        database_image.scale(0.4).shift(LEFT * 5.5)  # Adjust the scaling and position as needed

        # Display the image
        self.play(FadeIn(database_image))

        # Move the image to the middle of the table and shrink it
        self.wait(2)

        iris_image = ImageMobject("resources/images/iris.jpg")
        iris_image.scale(0.05).move_to(database_image.get_center())

        self.play(iris_image.animate.scale(4))
        self.play(iris_image.animate.move_to(efficient_net.get_center()).scale(0))

        self.wait(2)

        pixel_grid = PixelGrid()
        self.add(pixel_grid)
        pixel_grid.scale(0.05)
        pixel_grid.move_to(efficient_net.get_center())
        self.play(FadeIn(pixel_grid))

        self.play(pixel_grid.animate.scale(4).move_to(RIGHT*2))

        self.wait(2)


        diffusion_model = Rectangle
        self.add()