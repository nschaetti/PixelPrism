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
from pixelprism import find_animable_mixins
from pixelprism.animate import animeattr
from pixelprism.animate.able import AnimableMixin


# Non-animable object
class NonAnimableObject:
    def __init__(self):
        self.data = "I am not animable"
    # end __init__
# end NonAnimableObjectq


# Animation class with _find_animable_mixins already implemented as provided
class TestFindAnimableMixins(unittest.TestCase):

    def test_single_animable_object(self):
        """
        Test with a single animable object.
        """
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

        @animeattr('child')
        @animeclass
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

        @animeattr('inner')
        @animeclass
        class MiddleObject(AnimableMixin):
            def __init__(self):
                super().__init__()
                self.inner = DeepAnimable()
            # end __init__
        # end MiddleObject

        @animeattr('middle')
        @animeclass
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
        # end AnimableA

        class AnimableB(AnimableMixin):
            pass
        # end AnimableB

        @animeattr('a')
        @animeattr('b')
        @animeclass
        class ComplexObject:
            def __init__(self):
                self.a = AnimableA()
                self.b = AnimableB()
            # end __init__
        # end ComplexObject

        # Create an object with multiple animable objects
        obj = ComplexObject()

        # Test with multiple animable objects
        animable_objects = find_animable_mixins(obj)
        self.assertEqual(len(animable_objects), 2)
        self.assertIsInstance(animable_objects[0], AnimableMixin)
        self.assertIsInstance(animable_objects[1], AnimableMixin)
    # end test_multiple_animable_objects

    def test_with_non_animable_objects(self):
        """
        Test with a mix of animable and non-animable objects.
        """
        @animeattr('animable')
        @animeattr('nonanimable')
        @animeclass
        class MixedObject:
            def __init__(self):
                self.animable = AnimableMixin()
                self.non_animable = NonAnimableObject()
            # end __init__
        # end MixedObject

        # Create an object with a mix of animable and non-animable objects
        obj = MixedObject()

        # Test with a mix of animable and non-animable objects
        animable_objects = find_animable_mixins(obj)
        self.assertEqual(len(animable_objects), 1)
        self.assertIsInstance(animable_objects[0], AnimableMixin)
    # end test_with_non_animable_objects

    def test_no_animable_objects(self):
        """
        Test with no animable objects.
        """
        @animeattr('inner')
        @animeclass
        class NoAnimableObject:
            def __init__(self):
                self.inner = NonAnimableObject()
            # end __init__
        # end NoAnimableObject

        # Create an object with no animable objects
        obj = NoAnimableObject()

        # Test with no animable objects
        animable_objects = find_animable_mixins(obj)
        self.assertEqual(len(animable_objects), 0)
    # end test_no_animable_objects

    def test_recursive_animable_objects(self):
        """
        Test with recursive animable objects.
        """
        @animeattr('child')
        @animeclass
        class RecursiveAnimable(AnimableMixin):
            def __init__(self):
                super().__init__()
                self.child = self  # Circular reference to self
            # end __init__
        # end RecursiveAnimable

        # Create an object with recursive animable objects
        obj = RecursiveAnimable()

        # Test with recursive reference
        animable_objects = find_animable_mixins(obj)
        self.assertEqual(len(animable_objects), 1)  # Should only find one instance, avoiding recursion
    # end test_recursive_animable_objects

    def test_is_inspectable_and_get_inspectable_attributes(self):
        """
        Test if the class is inspectable and get the inspectable attributes.
        """
        @animeattr('children')
        @animeattr('mapping')
        @animeclass
        class TestObject:
            def __init__(self):
                self.children = [AnimableMixin(), AnimableMixin()]  # List of animables
                self.mapping = {
                    "a": AnimableMixin(),
                    "b": NonAnimableObject()  # Not animable
                }
            # end __init__
        # end TestObject

        # Check if the class is inspectable
        self.assertTrue(TestObject.is_animeclass())

        # Check the inspectable attributes
        inspectable_attrs = TestObject.animeclass_attributes()
        self.assertEqual(inspectable_attrs, ['mapping', 'children'])
    # end test_is_inspectable_and_get_inspectable_attributes

    def test_non_inspectable_class(self):
        """
        Test if the class is not inspectable.
        """
        class NonInspectableObject:
            def __init__(self):
                self.child = AnimableMixin()  # This will not be inspected
            # end __init__
        # end NonInspectableObject

        # Check if the class is not inspectable
        self.assertFalse(hasattr(NonInspectableObject, 'is_animeclass'))
        self.assertFalse(hasattr(NonInspectableObject, 'animeclass_attributes'))
    # end test_non_inspectable_class

# end TestFindAnimableMixins

if __name__ == '__main__':
    unittest.main()
# end if