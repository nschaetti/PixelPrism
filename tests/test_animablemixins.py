#
# This file is part of the Pixel Prism distribution (https://github.com/nschaetti/PixelPrism).
# Copyright (c) 2024 Nils Schaetti.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

# Imports
import unittest
from pixel_prism import find_animable_mixins
from pixel_prism.animate.able import AnimableMixin


class NonAnimableObject:
    def __init__(self):
        self.data = "I am not animable"
    # end __init__
# end NonAnimableObjectq


# Animation class with _find_animable_mixins already implemented as provided
class TestFindAnimableMixins(unittest.TestCase):

    def test_single_animable_object(self):
        class SingleAnimableObject(AnimableMixin):
            pass
        # end SingleAnimableObject

        # Create a single animable object
        obj = SingleAnimableObject()

        # Test with a single animable object
        animable_objects = find_animable_mixins(obj)

        # Check that the list contains the single animable object
        self.assertEqual(len(animable_objects), 1)
        self.assertIs(animable_objects[0], obj)
    # end test_single_animable_object

    def test_nested_animable_objects(self):
        """
        Test with nested animable objects.
        """
        class ChildAnimable(AnimableMixin):
            pass
        # end ChildAnimable

        class ParentObject:
            def __init__(self):
                self.child = ChildAnimable()
            # end __init__
        # end ParentObject

        # Create a parent object containing an animable child
        obj = ParentObject()

        # Test with a parent containing an animable child
        animable_objects = find_animable_mixins(obj)
        self.assertEqual(len(animable_objects), 1)
        self.assertIsInstance(animable_objects[0], ChildAnimable)
    # end test_nested_animable_objects

    def test_deeply_nested_animable_objects(self):
        """
        Test with deeply nested animable objects.
        """
        class DeepAnimable(AnimableMixin):
            pass
        # end DeepAnimable

        class MiddleObject(AnimableMixin):
            def __init__(self):
                super().__init__()
                self.inner = DeepAnimable()
            # end __init__
        # end MiddleObject

        class OuterObject(AnimableMixin):
            def __init__(self):
                super().__init__()
                self.middle = MiddleObject()
            # end __init__
        # end OuterObject

        # Create an object with deep nested animable objects
        obj = OuterObject()

        # Test with deep nested animable objects
        animable_objects = find_animable_mixins(obj)

        # Check that the list contains the deep animable object
        self.assertEqual(len(animable_objects), 3)
        self.assertIsInstance(animable_objects[0], OuterObject)
        self.assertIsInstance(animable_objects[1], MiddleObject)
        self.assertIsInstance(animable_objects[2], DeepAnimable)
    # end test_deeply_nested_animable_objects

    def test_multiple_animable_objects(self):
        """
        Test with multiple animable objects.
        """
        class AnimableA(AnimableMixin):
            pass

        class AnimableB(AnimableMixin):
            pass

        class ComplexObject:
            def __init__(self):
                self.a = AnimableA()
                self.b = AnimableB()

        obj = ComplexObject()
        animation = Animation()

        # Test with multiple animable objects
        animable_objects = animation._find_animable_mixins(obj)
        self.assertEqual(len(animable_objects), 2)
        self.assertIsInstance(animable_objects[0], AnimableMixin)
        self.assertIsInstance(animable_objects[1], AnimableMixin)

    def test_with_non_animable_objects(self):
        class MixedObject:
            def __init__(self):
                self.animable = AnimableMixin()
                self.non_animable = NonAnimableObject()

        obj = MixedObject()
        animation = Animation()

        # Test with a mix of animable and non-animable objects
        animable_objects = animation._find_animable_mixins(obj)
        self.assertEqual(len(animable_objects), 1)
        self.assertIsInstance(animable_objects[0], AnimableMixin)

    def test_no_animable_objects(self):
        class NoAnimableObject:
            def __init__(self):
                self.inner = NonAnimableObject()

        obj = NoAnimableObject()
        animation = Animation()

        # Test with no animable objects
        animable_objects = animation._find_animable_mixins(obj)
        self.assertEqual(len(animable_objects), 0)

    def test_recursive_animable_objects(self):
        class RecursiveAnimable(AnimableMixin):
            def __init__(self):
                super().__init__()
                self.child = self  # Circular reference to self

        obj = RecursiveAnimable()
        animation = Animation()

        # Test with recursive reference
        animable_objects = animation._find_animable_mixins(obj)
        self.assertEqual(len(animable_objects), 1)  # Should only find one instance, avoiding recursion


if __name__ == '__main__':
    unittest.main()
